# Watermark Anything With Localized Messages

- Decision: Accept (Poster)
- Scores: 3, 8, 8, 5, 5, 8

## Abstract
Image watermarking methods are not tailored to handle small watermarked areas.
This restricts applications in real-world scenarios where parts of the image may come from different sources or have been edited.
We introduce a deep-learning model for localized image watermarking, dubbed the Watermark Anything Model (WAM). 
The WAM embedder imperceptibly modifies the input image, while the extractor segments the received image into watermarked and non-watermarked areas and recovers one or several hidden messages from the areas found to be watermarked.
The models are jointly trained at low resolution and without perceptual constraints, then post-trained for imperceptibility and multiple watermarks.
Experiments show that WAM is competitive with state-of-the art methods in terms of imperceptibility and robustness, especially against inpainting and splicing, even on high-resolution images. 
Moreover, it offers new capabilities: WAM can locate watermarked areas in spliced images and extract distinct 32-bit messages with less than 1 bit error from multiple small regions -- no larger than 10\% of the image surface -- even for small $256\times 256$ images.
Training and inference code and model weights are available at https://github.com/facebookresearch/watermark-anything.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes a deep-learning-based image watermarking method, dubbed WAM. The encoder of WAM can watermark images with multiple watermarks/32-bit messages and the decoder of WAM can output a detection/segmentation map that identifies the regions containing watermarks and those that do not (called detection) and also a message map that presents the message in each watermarked area (called decoding).

### Strengths
1. The proposed method is able to embed multiple watermarks/32-bit messages
2. The method can output a detection/segmentation map that identifies the regions containing watermarks and those that do not

### Weaknesses
1. I am unclear on how the model achieves multiple watermark encodings. If I understand correctly, according to Figure 2, the model does not accept a mask input during the inference stage. In that case, how does it inject different messages into different regions? This makes Figure 1 somewhat misleading, as the first two steps suggest that the message can be precisely controlled to be encoded into specific objects. Could the authors provide further clarification on this point?

2. Could you clarify how the mask for the watermark's position is inputted into WAM in Figure 7 of the Appendix? Additionally, could you explain why this specific position was chosen for the localization experiment?

3. In lines 226-232, if I am not mistaken, this method appears to be identical to the method proposed by Bui et al., rather than merely 'a similar approach'. Please provide proper credit.

4. Regarding the claim in lines 304-305, it is unclear why applying JND would make training easier compared to using perceptual loss. Could the authors provide evidence or reasoning to support this claim?   'Applying JND only in the second training phase makes training easier than previous methods relying on “perceptual” or “contradictory” losses.'

5. I believe the reason WAM outperforms the baselines in terms of bit accuracy for splicing is simply because WAM incorporates splicing during training, whereas the other models do not. If the other models were to apply splicing as an augmentation or attack during training, would WAM still outperform them in terms of bit accuracy?

6. Typically, the TPR is reported at a fixed FPR, as seen in Tree Ring. Is the approach in this paper, reporting TPR at varying FPRs, fair and appropriate?

7. I am not entirely convinced of the necessity of embedding multiple messages separately into a single image. If the embedding capacity is large enough (e.g., 100 bits), users could already encode several different messages within a single bit stream. Could the authors provide some practical scenarios where embedding multiple separate messages would be particularly beneficial?

8. What happens if the position masks of multiple watermarks overlap? Additionally, for the watermark to be successfully extracted, what is the minimum proportion the watermark mask should cover relative to the overall image size? How many watermarks can be embedded in a single image at most?

9. A small typo on line 407, 'hide messages or arbitrary length' should be 'hide messages of arbitrary length'?

10. Will the source code and pretrained model be released for the community?

11. Is WAM robust to the new type of regeneration attacks [1], like diffusion-based regeneration?

---
[1] Invisible Image Watermarks Are Provably Removable Using Generative AI

---

I hope I have understood the method correctly. If there are any misunderstandings leading to wrong judgment, I apologize to the authors. I would be open to adjusting my score if the authors address my concerns during the rebuttal.

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper generalizes for multi-source images and achieves image locality preservation without a priori knowledge. The most innovative is to define watermarking as a segmentation task. Other contributions have been proposed in state-of-the-art papers, not for the first time in this thesis, and are not innovative enough.

### Strengths
The paper introduces the Watermark Anything Model (WAM), a deep learning-based approach for embedding and extracting localized image watermarks, which excels in efficiently embedding and locating watermarks within small areas, demonstrating resilience against image tampering and splicing. It can extract distinct 32-bit messages from multiple regions, each less than 10% of the image area, with minimal bit error rates. Trained at low resolutions and further optimized for perception invisibility and multiple watermarks, WAM competes with state-of-the-art methods in terms of imperceptibility and robustness, offering new possibilities and enhanced practicality for image watermarking technology.

### Weaknesses
1. Insufficient innovation, little change in the existing image watermarking framework, no analysis for specific modules, still using the traditional training architecture, just increase the template after random clipping, to achieve the application of the scene changes. 
2.Many existing work has been involved in local image watermarking, known as multi-source image watermarking [1]. The arbitrary image watermarking framework proposed in this paper though does not require prior knowledge and can be protected against any image. But only the major part of the image in a real scene has a role to play and there is no need to add watermarking information for any pixel.

[1] Wang G, Ma Z, Liu C, et al. MuST: Robust Image Watermarking for Multi-Source Tracing[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(6): 5364-5371.

### Questions
1. The paper compares EditGuard's work, but does not compare the work that is very similar to this paper [1], please explain the difference between the two;
2. How to guarantee the imperceptibility of the framework, and did not see the loss module in the mature watermarking framework, such as the MSE loss between the embedded image x_m and x, the loss of discriminative networks, etc.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the limitation of current image watermarking methods in handling small watermarked areas. The authors redefine the watermarking task as a segmentation problem, decomposing it into localization and detection tasks. They adopt a two-stage training strategy to separately achieve robustness and imperceptibility. By incorporating localization capabilities, the authors enable embedding multiple watermarks within a single image. Extensive experiments demonstrate the broad applicability of their method.

### Strengths
1. The paper introduces an innovative approach by transforming the watermarking task into a segmentation task, enabling the embedding of multiple watermarks at different locations in an image and supporting watermarking of small areas.
2. The proposed method shows broad applicability and can handle high-resolution images and images of various sizes.
3. The design motivations are well-justified, and the implementation details are clear. For improving visual quality, the authors opted for the JND approach instead of common "perceptual" or "contradictory" losses. Extensive experiments validate the effectiveness of using JND for watermarking.

### Weaknesses
1. Although the authors discuss different model architectures, the overall parameter size remains significantly larger compared to prior models such as HiDDeN (454.4K), FIN (737.8K) [2], and CIN (36.01M) [3]. While the authors highlight that their encoder (1.1M parameters) is relatively compact, it is still substantially larger compared to earlier models like HiDDeN (188.93K) and FIN (747.80K). This large parameter size may limit the method's applicability in resource-constrained environments.
2. The comparative experiments are not comprehensive. The authors do not consider more advanced watermarking models, such as MBRS, FIN, and CIN, which significantly outperform HiDDeN.
3. It would be helpful for the authors to provide a relationship graph between the extractor’s parameter size and performance, to justify the necessity of the large parameter size for detection and decoding tasks.

Overall, I am optimistic about the paper's method and the problem it addresses. I am particularly intrigued by the authors' segmentation-based approach for embedding multiple watermarks within a single image. My final score will depend on the authors' rebuttal.


[1] Fang H, Qiu Y, Chen K, et al. Flow-based robust watermarking with invertible noise layer for black-box distortions[C]//Proceedings of the AAAI conference on artificial intelligence. 2023, 37(4): 5054-5061.

[2] Ma R, Guo M, Hou Y, et al. Towards blind watermarking: Combining invertible and non-invertible mechanisms[C]//Proceedings of the 30th ACM International Conference on Multimedia. 2022: 1532-1542.

### Questions
See weaknesses.

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
4

### Summary
The paper presents a novel framework, WAM, for localized image watermarking that embeds and extracts watermarks from specific regions within an image. WAM’s key innovation is treating watermarking as a segmentation task, which enables the model to identify and decode watermarked regions even when only small parts of an image are modified. The model operates in two phases: initial training for robustness and a fine-tuning stage to ensure imperceptibility and support for multiple watermarks.

### Strengths
1. WAM redefines watermarking as a pixel-level segmentation task, allowing for precise localization and recovery of watermarks within small image regions. This advancement expands watermarking capabilities beyond global, image-wide methods.
2. WAM’s architecture combines an encoder-decoder structure and clustering techniques, making it well-suited for segmentation. The two-stage training, which introduces imperceptibility and multi-message support, adds flexibility and enhances usability in high-resolution and high-precision tasks.

### Weaknesses
1. Although the paper compares WAM with several baseline watermarking models, including advanced steganographic methods could provide a more comprehensive assessment of WAM's relative strengths and limitations. Additionally, embedding and extracting multiple watermarks in different regions of an image is not a unique concept, as it has been previously introduced in methods such as [a] and [b].
2. The need for pixel-level detection across high-resolution images could make WAM computationally intensive, especially in scenarios requiring real-time analysis. Further evaluation of computational costs would help contextualize its practical limitations.
3. The paper acknowledges WAM’s limited payload capacity (32 bits per region). While suitable for smaller messages, the model may face challenges in applications requiring larger payloads. Future work could explore optimizing the architecture to balance capacity and robustness.
4. The reliance on DBSCAN parameters for multi-watermark clustering introduces an element of complexity. Parameter tuning may be required across different datasets or conditions, potentially complicating deployment.

[a] Pan, Wenwen, Yanling Yin, Xinchao Wang, Yongcheng Jing, and Mingli Song. "Seek-and-hide: adversarial steganography via deep reinforcement learning." IEEE Transactions on Pattern Analysis and Machine Intelligence 44, no. 11 (2021): 7871-7884.
[b] Meng, Ruohan, Qi Cui, Zhili Zhoul, Chengsheng Yuan, and Xingming Sun. "A Novel Steganography Algorithm Based on Instance Segmentation." Computers, Materials & Continua 63, no. 1 (2020).

### Questions
Please clarify the inovitive of WAM accoding to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduced a new approach to image watermarking called the Watermark Anything Model (WAM). It allowed identify and localize the watermarked regions rather than just determined if an image is watermarked. 
WAM used a two-stage training process. In the first stage, based on the traditional LDM encoder-decoder architecture, the watermark messages were embedded into every single pixel of the image. Detection and Decoding were then measured by calculating the portion or weighted average of the detected watermarked pixels. The second stage applied the Just-Noticeable-Difference map to make the watermark more imperceptible. 
 This paper introduced a novel approach that allowed the localization of the watermarked regions; it proposed a two-stage training process rather than traditional DNN-based method that balanced between robustness and imperceptibility; It allowed multiple distinct watermarks to be embedded into a single image.

### Strengths
Originality:
WAM aimed to localize the watermarked region within an image, which is a novel perspective in watermarking. On the other hand, WAM was able to handle multiple distinct watermarks within a single image, which was rarely addressed in prior work. Additionally, WAM has a two-stage training process, first stage trained robustness while the second stage trained imperceptibility,  which is different from the traditional DNN-based methods. 

Quality:
The authors provided comprehensive experiments on standard datasets, which strengthened the validity of the results. 

Clarity:
The paper was well-structured and clearly explained. The use of toy examples and tables enhanced the readability and understanding of the paper.

Significance:
 The paper introduced watermark localization and a two-stage training process, which could open up new possibilities in watermarking technology.

### Weaknesses
1.	Limited capacity
WAM’s capacity was limited to 32 bits, which may be insufficient for many real-world applications. On the other hand, the comparison with the baseline method were not entirely fair when they are designed for larger capacity.
2.	Although utilized the JND map to improve watermark imperceptibility, it was only comparable to, but not outstanding among, baseline methods. On the other hand, since WAM used a two-stage training process, first training for robustness and then for imperceptibility, it was unclear how well WAM balanced the trade-off between robustness and imperceptibility when training these aspects separately.
3.	 The paper lacked discussion of the computational complexity, which would be valuable for understanding its efficiency and scalability in real-world applications.

### Questions
1.	In the two-stage training process, robustness and imperceptibility were trained separately.  Could the authors clarify how they ensure that this separation does not lead to conflict? For instance, robustness training led to visible artifacts, while imperceptibility training weakened robustness. How is this balance maintained effectively? 
2.	Could the authors discuss the trade-off between the bit capacity, robustness, imperceptibility? It is well-known that there is an inherent trade-off among these three dimensions in watermarking. In this paper, bit capacity was sacrificed, yet imperceptibility was not outstanding than the baseline, and robustness comparisons were limited to a specific scenario: localized watermarked regions. 
3.	While WAM used a JNP map to enhance imperceptibility, its imperceptibility was comparable but not outstanding relative to baseline methods. Is there potential for further improvement in imperceptibility without compromising robustness?
4.	Given that the capacity of WAM was limited, its ability to support multiple distinct watermarks is constrained. As the amount of AIGC grows rapidly, do the authors have suggestions for extending WAM to support larger payload?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new method, Watermark Anything Model (WAM), for adding watermarks to images. The method demonstrates state-of-the-art performance even under various manipulations, such as splicing, geometric transformations, color changes, and inpainting. It consists of an embedder and an extractor capable of localizing the watermark region and decoding distinct messages in different areas of the image. The paper is well-written and easy to understand. Overall, the proposed approach offers a robust solution for image watermarking, with potential applications across diverse digital media fields.

### Strengths
The method has been tested against various manipulations, showcasing its robustness. It reliably detects watermarks even in image splicing and inpainting, including small or detailed areas. The paper is well-structured, making it clear, easy to read, and accessible to the reader.

### Weaknesses
The watermark model has not yet been tested against attacks from generative AI, which is widely used today. The paper would be strengthened if the watermark remains detectable after processing with super-resolution models (whether CNN-based or Stable Diffusion-based). Otherwise, people could easily remove the watermark without much effort. The model demonstrates a 100% true positive rate in many cases; however, most of these manipulations are included in the training augmentation, which may indicate overfitting rather than true robustness. If a manipulation is not included in the training data, can the watermark still be detected? Please also elaborate on the model performance against VAE attack and diffusion attack.

### Questions
In the paper, the authors talk about generative AI models, but they do not show how strong the model is against these types of AI, like Stable Diffusion, which is popular in the AI art community. With Stable Diffusion, it’s possible that all the hidden messages could disappear when the image is changed. It’s important to show that the model can handle generative AI modifications, like Stable Diffusion. For example, if I use Stable Diffusion-based super-resolution (a common tool for improving image quality) on the image, can the model still detect the watermark? If not, the watermark becomes easy to remove.

Another useful test would be to see if adding a second watermark to an image removes or covers the first one. For instance, after editing an image, if I use WAM again to add my own watermark, can the original watermark message still be found?

Please do these tests to show how strong the model is, or suggest ways to prevent these possible problems.

### Soundness
3

### Presentation
3

### Contribution
3
