# Data Overfitting for On-Device Super-Resolution with Dynamic Algorithm and Compiler Co-Design

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 5, 5

## Abstract
Deep neural networks (DNNs) are frequently employed in a variety of computer vision applications. Nowadays, an emerging trend in the current video distribution system is to take the advantage of DNNs overfitting property to perform video resolution upscaling. By splitting videos into chunks and applying a super-resolution (SR) model to overfit each chunk, this scheme of SR models plus video chunks is able to replace traditional video transmission to enhance video quality and transmission efficiency. However, many models and chunks are needed to guarantee a high performance, which leads to tremendous overhead on model switching and memory footprints at the user end. To resolve such problems, we propose a Dynamic Deep neural network assisted by a Content-Aware data processing pipeline to reduce the model number down to one (Dy-DCA), which helps promote performance while conserving computational resources. Additionally, to achieve real acceleration on the user end, we design a framework that optimizes dynamic features (e.g., dynamic shapes, sizes, and control flow) in Dy-DCA to enable a series of compilation optimizations, including fused code generation, static execution planning, etc. By employ such techniques, our method achieves better PSNR and real-time performance (33 FPS) on an off-the-shelf mobile phone. Meanwhile, assisted by our compilation optimization, we achieve 1.7$\times$ speedup while saving up to 1.61$\times$ memory consumption.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel framework, Dy-DCA, which combines a Dynamic Deep Neural Network with a Content-Aware data processing pipeline to enhance on-device super-resolution for videos. This approach aims to address the challenges of model switching overhead and memory footprint associated with traditional video super-resolution methods that rely on splitting videos into chunks and overfitting a super-resolution model to each chunk.

**Key Contributions:**
- Dynamic Neural Network with Content-Aware Data Processing: The paper proposes a scalable dynamic deep neural network paired with a fine-grained data processing method, significantly reducing the number of required models while maintaining high performance and reasonable model size. This is achieved by dynamically producing patches of different texture complexity and overfitting these patches with a designed dynamic neural network.

- Compiler-Level Optimization: To accommodate the dynamic nature of the proposed neural network and to ensure real-time performance on devices, the paper introduces a compiler-level optimization framework. This framework optimizes dynamic features such as input shapes and control flow, enabling a series of compilation optimizations that result in faster execution and reduced memory consumption.

- Enhanced Video Quality and Efficiency: By employing the proposed framework and optimizations, the paper claims to achieve better PSNR and real-time performance on mobile devices, along with significant improvements in speed (1.7× overall speedup) and memory consumption (up to 1.61× savings).

### Strengths
**Originality:**
The paper introduces a novel framework, Dy-DCA, which combines a dynamic deep neural network with a content-aware data processing pipeline for on-device super-resolution. This approach is original as it addresses the common issue of model switching overhead in video super-resolution, reducing the number of required models to one. The integration of a compiler-level optimization framework to support the dynamic nature of the neural network adds a unique dimension to the work, showcasing an innovative solution to a well-known problem in video super-resolution.

**Quality:**
The paper appears to be of high quality, providing a comprehensive and well-thought-out solution to enhance video quality and efficiency in transmission. The proposed Dy-DCA framework and the associated compiler-level optimizations are grounded in solid theoretical and practical considerations, with claims of significant improvements in PSNR, real-time performance on mobile devices, and resource efficiency.

**Clarity:**
The paper is well-structured and articulates the problem, proposed solution, and contributions clearly. The use of figures and step-by-step explanations aid in understanding the complex concepts involved in the Dy-DCA framework and compiler-level optimizations. However, the depth of the content may require readers to have a substantial background in deep learning, computer vision, and compiler optimizations.

**Significance:**
The paper holds significant potential to impact the field of video streaming and super-resolution, addressing critical issues of video quality and transmission efficiency on edge devices. By reducing model switching overhead and memory footprint, the paper presents a solution that could lead to more efficient and effective video super-resolution applications, especially in real-time scenarios on resource-constrained devices.

### Weaknesses
- It would be more comprehensive to add ablation studies to understand the contribution of each component of the Dy-DCA framework and the compiler-level optimizations would offer a clearer picture of their individual and combined effects on performance and quality.

- The paper could benefit from experiments to evaluate the scalability and efficiency of the proposed framework on a variety of hardware architectures, including both high-end and low-end devices, would provide a comprehensive view of its applicability and performance across different scenarios.

- The PSNR gain is marginal, though the speedup is shiny.

### Questions
Could you clarify how the PSNR gain in table 2 and 4 are *significant*? From Figure 4, visually the performance seems close to other approaches.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a Dynamic Deep neural network assisted by a Content-Aware data processing pipeline to reduce the number of models down to one (Dy-DCA), while still maintaining good performance. Meanwhile, the paper designs a framework that optimizes dynamic features (e.g., dynamic shapes, sizes, and control flow) in Dy-DCA to enable a series of compilation optimizations, including fused code generation, static execution planning, etc. The performance of the proposed solution is evaluated on two datasets.

### Strengths
1. The proposed solution achieves better PSNR and real-time performance compared to traditional video transmission, while reducing the number of models needed for high performance down to one and conserving computational resources.

2. The proposed solution optimizes dynamic features (such as dynamic shapes, sizes, and control flow) to enable a series of compilation optimizations (including fused code generation, static execution planning, etc.), which helps achieve acceleration on the user end.

### Weaknesses
1. The paper lacks a detailed discussion of the limitations and potential drawbacks of the proposed solution.

2. The paper does not provide a detailed comparison of the proposed solution with other state-of-the-art methods in terms of model size and computational complexity.

3. The paper only provide PSNR performance. Additional subjective quality (such as MS-SSIM) should be measured.

4. The paper does not provide ablation experiments.

### Questions
Please refer to weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes dynamic processing of different regions in video frames. Furthermore, the authors propose a compiler that manages the dynamism of tensor shapes.

### Strengths
Real-time video super resolution on a mobile device at 1080p is an impressive result.

### Weaknesses
Rewriting the paper to be more accessible to non-expert readers would be beneficial. As it stands, the flow of information assumes the reader is familiar with compilers, the current state of SR, and the continuous model adaptation framework, which may not always be the case.

### Questions
1. The paper rightly points out that switching the Super-Resolution (SR) model from one chunk to another can be costly. Could the solutions proposing sparse incremental model changes address this issue (refer to [1] for an example)?

2. The unique contributions in Section 2.3 are difficult to comprehend in their current form. How does this categorization of tensors encompass all use cases? What are the limitations of the proposed compiler? Is it applicable solely to SR or also to convolutional models?


[1] Khani, Mehrdad, Vibhaalakshmi Sivaraman, and Mohammad Alizadeh. "Efficient video compression via content-adaptive super-resolution." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4521-4530. 2021.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Deep neural networks (DNNs) are increasingly utilized for video resolution upscaling in modern video distribution systems, enhancing quality and efficiency by overfitting each video chunk with super-resolution (SR) models. However, the high number of models and chunks required causes substantial overhead in model switching and memory usage. To tackle this, a new method, Dy-DCA, employs a single dynamic deep neural network with a content-aware pipeline, significantly reducing the model count. This approach not only improves performance and quality (measured by PSNR) on standard mobile devices but also achieves a 1.7× speedup and up to 1.61× memory savings.

### Strengths
* Reducing one dynamic DNN to minimizing the switching overhead is clever.
* Significant improvement in FPS on off-the-shelf mobile phones while maintaining PSNR quality.

### Weaknesses
* The presentation of the paper, particularly in Section 2.2 and Section 2.3, needs significant improvement due to a lack of details.

### Questions
This is more of a general comment than a specific question. The on-device super-resolution application presented in this work is intriguing, as it enhances video quality using a single dynamic DNN and achieves high FPS through model-compiler co-design. While I understand the constraints of the page limit, the current description of the proposed system (incl. Sections 2.3.3 and 2.3.4) is overly terse and challenging to comprehend for a non-expert reviewer. Therefore, a substantial revision is required to improve the presentation. There are also typos, which requires more careful proofreading.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
