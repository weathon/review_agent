# EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Robust 3D hand reconstruction is challenging in egocentric vision due to depth ambiguity, self-occlusion, and complex hand-object interactions. Prior works attempt to mitigate the challenges by scaling up training data or incorporating auxiliary cues, often falling short of effectively handling unseen contexts. In this paper, we introduce EgoHandICL, the first in-context learning (ICL) framework for 3D hand reconstruction that achieves strong semantic alignment, visual consistency, and robustness under challenging egocentric conditions. Specifically, we develop (i) complementary exemplar retrieval strategies guided by vision–language models (VLMs), (ii) an ICL-tailored tokenizer that integrates multimodal context, and (iii) a Masked Autoencoders (MAE)-based architecture trained with 3D hand–guided geometric and perceptual objectives. By conducting comprehensive experiments on the ARCTIC and EgoExo4D benchmarks, our EgoHandICL consistently demonstrates significant improvements over state-of-the-art 3D hand reconstruction methods. We further show EgoHandICL’s applicability by testing it on real-world egocentric cases and integrating it with EgoVLMs to enhance their hand–object interaction reasoning. Our code and data will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the first in-context learning framework (EgoHandICL) for 3D hand reconstruction under challenging egocentric conditions. EgoHandICL consists of a VLM-based exemplar retrieval strategy, a multi-modal feature tokenizer, and a Masked Autoencoders (MAE)-based architecture for hand mesh reconstruction. The proposed framework achieves significant performance improvement over state-of-the-art 3D hand reconstruction models on two public benchmarks of ARCTIC and EgoExo4D.

### Strengths
1) The in-context learning framework for 3D hand construction is novel. It augments existing hand reconstruction models from an interesting perspective. The motivation is clear, and the method design based on VLM and masked autoencoder is also insightful.  
2) Comprehensive experiments are conducted to analyze the performance of the proposed framework which significantly improve the performance over SOTA methods. 
2) The paper is well presented and it is easy to follow the story.

### Weaknesses
1) Overall, the paper is clearly written. However, several technical details are still missing.   
a) For the ICL tokenizer part, it is not clear how each set of the ICL tokens are composed of (e.g., number of tokens, dimension of token embedding).   
b) The detailed architecture of MANO encoder/decoder and Transformer-based reconstruction model are not given.  
c) In Section 4.2, it is claimed that "Our default setting balances this trade-off by applying the reasoning prompts under heavy occlusion and the description prompts in clearer scenarios". However, it is unclear about the implementation details, such as how to quantify occlusion or scene clearness.  
2) Several experiments are missing, which are important to demonstrate the overall performance of the proposed framework:  
a) The ablation study for the ICL tokenizer (especially the cross-attention to fuse multi-modal tokens) is not given.  
b) In Section 3.3, the loss fuctions require ground-truth MANO parameters as well as 3D mesh/vertex. However, such information are not provided in the EgoExo4D dataset. Then, how the proposed method is trained on EgoExo4D?  
c) It is also important to show the runtime evaluation. In particular, as explained in Appendix B.1, the retrieval is repeated 3 times for visual template. Considering the large size of training data, the template retrieval would be very time-consuming.

### Questions
The questions are listed above in the Weakness part. Overall, the paper proposes a novel method and the experiments are extensive and convincing. My positive rating to this paper would be solidified if the above concerns are addressed.

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
This paper tackles 3D hand reconstruction from egocentric views by applying in-context learning (ICL) paradigm, named EgoHandICL. Given a query image, a Vision-Language Model (VLM) is prompted to retrieve a single, contextually relevant template image from a database. The core EgoHandICL model, a Masked Autoencoder (MAE)-based Transformer, receives multimodal tokens (representing image features, textual descriptions, and structural MANO parameters) for four components: the template input (a coarse 3D estimate), the template target (its ground truth), and the query input (its coarse 3D estimate) . The model is trained to predict the query target. Experiments on the ARCTIC and EgoExo4D benchmarks show that EgoHandICL significantly outperforms state-of-the-art methods like HaMeR and WiLoR. Furthermore, the ablation studies, regarding hand types, prompt designs, masking, and losses, confirm the effectiveness of the proposed module of the method.

### Strengths
- Novelty: The application of in-context learning to 3D hand reconstruction is highly novel. On top of the state-of-the-art pose estimators, it provides dynamic, example-based reasoning at inference time. Using a VLM for semantic retrieval (e.g., finding similar interactions or occlusion types ) rather than just visual similarity is a powerful and unique idea for this problem.
- Technical soundness: The method is well-formulated. Refining a corse prediction with exemplar mano pair is a logical and effective approach. Plus, the multimodal ICL tokenizer and MAE-style learning are clearly aligned to this problem and seem to be a strong prior for ICL in visual and 3D hand parameter domains. 
- Strong experimental support: The results are state-of-the-art across the board. The improvements are particularly large in highly occluded scenarios. Thorough ablation studies also strengthen the paper's argument.

### Weaknesses
Further clarifications on the following points would be appreciated. 
- Inference cost trade-off: The proposed method introduces significant computational overhead (VLM text generation, template retrieval, multimodal tokenization, and MAE transformer inference) beyond the base regressor. An estimation of this additional burden, using metrics like FPS or FLOPs, is needed to provide a comprehensive view of the test-time bottleneck for potential real-time applications.
- Impact of template size (N): The framework is evaluated using only a single template (N=1). Drawing an analogy from LLMs, an ablation study on using additional context (N>1) would be an insightful exploration to understand how context quantity benefits this task.
- Missing failure analysis: A discussion on the effect of noisy or poorly-matched exemplars is needed. For instance, out-of-distribution (non-typical) test samples may retrieve poor exemplars due to a lack of similar data. It would be insightful to know if this situation worsens the reconstruction performance compared to the baseline, which would clarify the framework's robustness.

### Questions
In addition to the Weaknesses section, please clarify points below.
- Is the MANO encoder and decoder architecture a simple, continuous autoencoder? Or does it implement a discrete latent variable, similar to a VQ-VAE?
- Need the details for visual prompts in EgoVLMs: How are the hand reconstructions used as visual prompts for EgoVLMs? Does this involve simply overlaying the predicted 3D hand mesh sequence onto the input video, or were other prompt variants tested?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors proposed a solution to 3D hand reconstruction from egocentric RGB images, which is challenging due to depth ambiguity, severe self-occlusion, and complicated hand-object interactions. The core idea is to bring in-context learning into the problem. The method first retrieves a template egocentric image that is semantically and visually similar, using VLMs. Then, for that template, authors construct an exemplar pair: coarse MANO estimate from a pretrained reconstructor and ground-truth MANO parameters for that template. They do the same for the query frame. The model is then conditioned on the paired examples to refine the query’s coarse MANO estimate into a final high-quality 3D hand mesh prediction, leveraging retrieved context without explicit fine-tuning at test time. Authors evaluate the method on ARCTIC and EgoExo4D datasets, and show that their reconstructions improve fine-grained hand-object reasoning.

### Strengths
Clear motivation: egocentric hand reconstruction is challenging. The paper successfully pointed out and tackled the challenges  in the problem and provided sound solutions.

Novel and sound idea: The retrieval and in-context learning are novel and sound approaches. Instead of simply training a bigger network or adding more auxiliary cues, authors explicitly retrieve semantically similar examples and feed them as context to guide inference. Especially, instead of simply retrieving raw RGB crops, authors proposed to align coarse prediction and GT pairs and let the model learn how to correct errors in the query. This seems an interesting and effective idea. Multimodal tokenizer also looks effective.

Strong empirical performance: Authors showed that injecting the reconstructed hands into the egocentric VLMs improves its ability to reason about fine-grained HOIs. This looks like a good downstream example.

### Weaknesses
The method may depend on the quality of retrieval: Even though the overall performance might depend on the quality of template, the paper does not quantify the success/failure of the retrieval stage. It might be better to include such results.

More careful analysis required to validate the effectiveness: Deep learning models are frequently suffering from out-of-distribution (OOD) samples, which denote testing samples far from training samples which are exploited during training. Using the retrieval scheme, it might be even hard to secure similar samples for all the testing samples in the limited database, which may worsen the overall accuracy. It might be necessary to analyze the relationship between average image similarity during testing and overall accuracy.

Efficiency issue: Since the method is based on VLMs, it might be hard to deploy the algorithm in the real-time manner.

### Questions
It might be useful to report how large the exemplar pool is. 
Please provide failure cases, especially for the retrieval stage.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EgoHandICL, an in-context learning (ICL)-based framework for 3D hand reconstruction in egocentric visual settings. It focuses on handling unseen contexts. Especially, it reformulates 3D hand reconstruction as exemplar retrieval for contextual adaptation and multimodal (image, MANO parameter, text ) token fusion based on MAE-style VLM for robust representation learning. Experiments on ARCTIC and EgoExo4D benchmarks show significant improvements over previous works (HaMeR, WiLoR, WildHand, and HaWoR). The paper further explores downstream integration with EgoVLMs, showing that reconstructed hand cues enhance hand–object interaction reasoning.

### Strengths
- The proposed method is an effective method applying in‑context learning (ICL) in 3D hand reconstruction. The retrieval and then multimodal learning strategy is sound and provides an interesting direction to contextual adaptation when interpreting complex egocentric scenes. 

- The paper is well structured and easy to follow. 

- Quantitative results on ARCTIC and EgoExo4D clearly demonstrate the advantage of EgoHandICL, including bimanual and occlusion-heavy cases. Ablations cover mask ratios, loss combinations, backbone variations, and prompting strategies, substantiating design choices. Integration with EgoVLMs (Tab.8, Fig.5) shows meaningful downstream benefits beyond reconstruction itself.

### Weaknesses
1. The retrieval is confusing and not clear, especially the definition of template/visual images. Based on "few shot" (L.154) and Fig.2, it seems that the template images are from the same dataset or even the frames of the same video clip. In this case, it seems the retrieval is not necessary and using template images is easier. Moreover, it would be better to discuss the influence if the retrieval/prompt is not good enough.

2. The proposed method takes many pretrained/foundational models. Eq.2 requires coarse MANO parameters from HaMeR or WiLoR, Eq.4 requires the pretrained Vit (WiLoR, L.212 and L.278), Eq.5 requires the pretrained 3D feature encoder Uni3D-ti (L.269 and L.281). Moreover, the backbone is Qwen2.5-VL-72B-Instruct and Qwen-7B (L.278). This raises concerns about efficiency. It would be better to report the latency, parameters, inference time (ms), FLOPs and GPU memory compared to other sotas. Moreover, it would be better to report the efficiency of each step in the proposed method.

3. The training details are not clear. It would be better to explicitly state the dataset used for training (Eq.4). In addition, I am not sure if the comparison is fair. For Tabs.1 and 2, are ARCTIC and EgoExo4D used during finetuning for all sotas? It would be better to show the training dataset of all methods.

4. More ablations are needed. The paper mentions a cross-attention fusion among visual, text, and MANO tokens, but does not ablate their individual contributions. It would be better to conduct an ablation study to evaluate performance without each modality (Image-only, Image+Text, Image+Structure) and quantify the benefit of multimodal fusion. Moreover, it would be better to show the performance after removing/reducing quality such as template, text, $M_{qry}$, $M_{tpl}$. 

5. The details of data preprocessing and hand-crafted prompts are not clear. This makes it difficult to reproduce the results faithfully.

6. While the experiments show that ICL improves occlusion reasoning, the paper lacks analysis explaining why ICL achieves this. A qualitative visualization or attention-weight inspection between exemplar templates and query tokens could help substantiate the claimed “context reasoning” mechanism.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
