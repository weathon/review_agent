# InsertNeRF: Instilling Generalizability into NeRF with HyperNet Modules

- Decision: Accept (poster)
- Scores: 10, 6, 6

## Abstract
Generalizing Neural Radiance Fields (NeRF) to new scenes is a significant challenge that existing approaches struggle to address without extensive modifications to vanilla NeRF framework. We introduce **InsertNeRF**, a method for **INS**tilling g**E**ne**R**alizabili**T**y into **NeRF**. By utilizing multiple plug-and-play HyperNet modules, InsertNeRF dynamically tailors NeRF's weights to specific reference scenes, transforming multi-scale sampling-aware features into scene-specific representations. This novel design allows for more accurate and efficient representations of complex appearances and geometries. Experiments show that this method not only achieves superior generalization performance but also provides a flexible  pathway for integration with other NeRF-like systems, even in sparse input settings. 
Code will be available at: https://github.com/bbbbby-99/InsertNeRF.

## Human Reviews

## Human Reviewer 1

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a method called InsertNeRF, which aims to instill generalizability into Neural Radiance Fields (NeRF) without extensive modifications to the vanilla NeRF framework. The method utilizes multiple plug-and-play HyperNet modules to dynamically tailor NeRF's weights to specific reference scenes, allowing for more accurate and efficient representations of complex appearances and geometries. The main contributions of InsertNeRF are: (1) introducing a novel paradigm that inserts HyperNet modules into NeRF-like systems to achieve generalizability, (2) designing two types of HyperNet module structures tailored to different NeRF attributes, and (3) demonstrating state-of-the-art performance and potential in various NeRF-like systems. The paper also discusses related works on generalizable NeRF and hyper-networks, and provides an overview of the method's background and implementation details.

### Strengths
-The idea of using HyperNet to solve the important problem of generalizable NeRF is a very good and promising pipeline. Thus, the novelty of the proposed method is very strong. 
-The proposed InsertNeRF can achieve the generalizability without extensive modifications to the vanilla NeRF framework. 
-HyperNet modules can dynamically tailor NeRF's weights to specific reference scenes, allowing for more accurate and efficient representations. 
- The proposed InsertNeRF achieves superior generalization performance and can be integrated with other NeRF-like systems. It also demonstrates state-of-the-art performance and potential in various NeRF-like systems, even in sparse input settings.
- The writing and the presentation of the paper is good.

### Weaknesses
-All the contributions, points, and the details of methods and experiments have been clearly presented.

### Questions
-How about the computational cost and efficiency of the proposed method compared with other methods?
- It is suggest to public their Source code for the readers to better reproduce the proposed method.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In the paper authors use multiple plug-and-play hypernetworks modules in NeRF based models to obtain generalization properties. The proposed method can be integrate with many different NeRF-based models.

### Strengths
1. The papers obtain good experimental results.
2. The framework can be used for many different NeRF architectures

### Weaknesses
1. The paper is hard to read. It is unclear what is input to new meeting components. 
2. It is hard to understand the general idea of the model, and Fig. 2 is completely unclear.
3. Genera formulas (3) and (4) are unceler.
4. In the experimental section we do not have experiments with the ShapeNet-based dataset (see pixelNeRF).

### Questions
1. The model use hypenetwork so shoud be compared with hypenetworks based method like Points2NeRF or HyperNeRFGAN
2. In Figure 1, the authors show that the introduced method works nicely with many NeRF framework but do not compare with other NeRF with generalization properties. It is misleading.
3. Section 2.1 do not include Hypernetworks based NeRF. Authors do not mention TriPlaneNeRF or MultiPlaneNERF. Furthermore the generative models using NeRF should be mentioned. 
4. It is quite difficult to understand formula (3). F_view take as an input one point (x,d) or all points on the ray? F_sample take as an input one point (x,d) or all points on the ray?
4. It is quite difficult to understand formula (4). it looks like  F_sample was change to NyperNet. What is an input to Hypernetwork and what is an input to F_sample? Such formula should be describe more carefully. 
5. Figure 2 is completely unclear.
6. Authors write, "While this method can be adapted to a variety of NeRF-based systems (Sec. 4.4), we focus on its application on the vanilla NeRF in this section." and then directly built the method on formula (3) dedicated to Generalizable NeRF.
7. How difficult is to find parameters lambda_1 and lambda_2 in const function?
8. In experimental section authors shud add experiments on ShapeNet data similar to pixelNeRF.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper targets the task of NeRF generalization, and proposes the HyperNet module before each MLP layer of vanilla NeRF, inspired by the concept of hypernetwork. The proposed method achieves comparable or even better performance compared to previous works.

### Strengths
Such a plug-and-play hypernetwork-based method can bring an extra performance boost for NeRF generalization, which is proved to be useful.

Comparison to previous works and several ablation studies were performed to verify the effectiveness of the aggregation strategy and each component proposed in the HyperNet module.

### Weaknesses
The paper may need more explanations or descriptions about the insight of the method, to demonstrate why such a strategy or technical designs could help NeRF generalization—the analysis of ``why'' is also important. 

The experiments are very simple and short, in Figure 3, the methods are close in visualizations. As a prior and baseline work of generalizable work, IBRNet shows close performance to the proposed InsertNeRF.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
