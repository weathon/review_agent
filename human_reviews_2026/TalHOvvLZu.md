# VAT: Vision Action Transformer by Unlocking Full Representation of ViT

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 4, 0

## Abstract
In robot learning, Vision Transformers (ViTs) are standard for visual perception, yet most methods discard valuable information by using only the final layer's features. We argue this provides an insufficient representation and propose the Vision Action Transformer (VAT), a novel architecture that is extended from ViT and unlocks the full feature hierarchy of ViT. VAT processes specialized action tokens with visual features across all transformer layers, enabling a deep and progressive fusion of perception and action generation. On a suite of simulated manipulation tasks, VAT achieves a 98.15\% average success rate across four LIBERO benchmarks, establishing a new state-of-the-art by outperforming prior methods like OpenVLA-OFT. Our work presents not only a powerful model for imitation learning but also demonstrates the critical importance of leveraging the complete "representation trajectory" of vision models to advance robotic policy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce Vision Action Transformer (VAT), a policy learning architecture that extends Vision Transformers (ViTs) by incorporating action tokens into the transformer sequence using cross-attention. Unlike conventional vision-language-action (VLA) models that often rely on the final layer of a pretrained vision backbone, VAT exploits the full hierarchy of visual representations across all layers, aiming to preserve more fine-grained details of the scene. The model is evaluated on the LIBERO benchmark suite and authors demonstrate a 98.15% success rate, outperforming state-of-the-art models such as OpenVLA-OFT. An empirical ablation on the importance of early layers is presented, further suggesting the importance of utilizing entire feature trajectories for downstream performance.

### Strengths
-	Improving our understanding of how to utilize foundation models for policy learning is a highly relevant research question and this paper tackles an interesting aspect of architectural design. Using a cross-attention mechanism to integrate multiple layers seems to be a sound way of closely integrating pretrained vision models with action prediction heads.
-	The paper carefully ablates multiple design choices, such as number of camera views, choice of ViT backbone (DinoV2, SigLIP2) and relative importance of representation depth for policy learning.
-	The baselines in Table 3 adequately represent the state of the art on the LIBERO benchmark and fall into a similar category of the proposed model architecture.
-	The VAT architecture is clearly explained through the use of Figure 1, as well as through formulas (1) to (8).
-	The success rate of 98.15% on LIBERO is strong in principle, but the setup is not entirely clear (see weaknesses & questions).

### Weaknesses
-	The presentation of tables, figures and formulas could be improved. Formulas (3) to (8) take up too much space. Table 1, Table 2 are ablations, yet appear before the methods section. Similarly, Table 3 appears before the methods section. Figure 2 is another ablation but could be reduced to a small table or small figure.
-	The authors only evaluate their approach on the LIBERO benchmark. To the best of my knowledge, it is common practice to evaluate on at least two benchmarks to ensure generality of the proposed approach. For example, I would suggest to include Calvin [1]) or a real world experiment.
-	At the 95-100% level of performance on LIBERO, performance is highly saturated. The paper does not seem to include any information about standard deviation or number of rollouts/seeds over which averaging is performed to obtain the claimed number, making the result unreliable.
- The paper seems to question the necessity of a pretrained language model component in VLA models. This is an interesting research question in itself, but it is not sufficiently addressed in this paper, making conclusive comparisons with the VLA baselines difficult. It is unclear whether utilizing multiple layers of the vision backbone benefits only the proposed VAT model on LIBERO, or if it is a general phenomenon. I believe that the paper would greatly benefit from broadening its horizon to specifically try to answer this question.

[1] Mees, Oier, et al. "Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks." IEEE Robotics and Automation Letters 7.3 (2022): 7327-7334.

### Questions
-	Could you elaborate on the evaluation settings used, such as number of runs and observed standard deviations?
-	OpenVLA combines SigLIP and DinoV2 representations, whereas VAT relies on a single model. What if you want to combine different vision backbones in your model?
-	Aside from utilizing multiple layers, this work seems to claim that the large language model component is unnecessary in VLAs. How can you expect to generalize to novel task descriptions without any language pretraining?

I would be happy to raise my score if the questions and weaknesses are addressed properly.

### Soundness
2

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
4

### Summary
This paper introduces the Vision Action Transformer (VAT), a robot imitation learning architecture that leverages the full hierarchy of representations within a Vision Transformer (ViT) backbone. Instead of relying solely on final-layer visual features, VAT injects action tokens alongside vision tokens and processes them concurrently across all transformer layers through cross-attention. Experiments on the LIBERO benchmark suite demonstrate strong performance. Additional ablations include layer-skipping experiments that demonstrate the utility of early-layer features, as well as attention visualizations that illustrate evolving focus patterns during perception-action fusion.

### Strengths
Strengths:
- The underutilization of intermediate ViT representations for robotic policy learning makes sense to utilize richer representations.
- Introducing action tokens across all ViT layers enables progressive perception-action coupling without manual layer selection - this is a conceptually simple idea. The simulation results show the effectiveness of it.
- Achieves 98.15% mean success rate on LIBERO and outperforms strong baselines such as OpenVLA-OFT.
- Offers both ablation and visualizations to help understand the contributions.

### Weaknesses
Weakness:
- All results are in simulation on one benchmark. It's not clear whether this method will transfer to a real-world robot setting. Considering that this is the main application domain of this paper, I feel that this is very necessary to convince the audience.
- There are no runtime, memory, or scalability comparisons versus alternatives (e.g., external/internal fusion or frozen ViT baselines). I assume that the proposed method will require more resources, but there is no substantial discussion on this.
- This one is related to the above one. There is no study of token scaling or capacity limits.  The number and structure of action tokens are fixed. Potential bottlenecks or optimality are not explored.
- There is a potential for overclaiming simplicity and efficiency. While design is conceptually simple, training all layers with added tokens could challenge the “parameter-efficient” claim.

### Questions
Please address the weakness points.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a new architecture for learning robot policies by extending the Vision Transformer (ViT) architecture called Vision Action Transformer (VAT). The authors claim that using features from the final layers of the vision transformer are generally not enough for robot policies and therefore they introduce a mechanism to incorporate features from all the layers in the ViT. The authors append special action tokens to the sequence of input vision tokens which are processed by special Action Modules that run parallel to each block in the ViT. These action tokens cross attend to visual features from the ViT blocks incorporating visual features from all the layers. The paper presents results on Libero10 and shows that VAT performs better than SOTA transformer based policies.

### Strengths
1. The paper presents a super simple idea for extending standard ViT into robot policies by using action modules parall to ViT blocks.  
2. The paper is very well written. 
3. The paper shows good performance improvement on the Libero benchmark compared to other state of the art methods.

### Weaknesses
1. Is it really about features from lower layers?

The paper motivates the use of parallel action modules by lack of finegrained details in the features of the final layer of the ViT. However, the action module is also sequential and the final action prediction only depends on the features from the last action module. My question is why would the action module retain any of the lower layer features and the ViT would not? 

This could be understood better by probing the final layer features of ViT for the “details” that the authors think might be missing and seeing if those details are better retained through the addition of action modules.

2. No Ablations?

Building on the above point, it would have been nice to see more ablations in the paper to help understand what actually is happening. 

A. Is the addition of a separate action module with cross attention really necessary? Would just addition of action tokens akin to CLS token in ViT suffice?

B. Effect of FiLM layers on the final performance? I think this is really important as one of the reasons VAT might work better is that the FiLM layers might be helping ground the model better to the given language task. VLAs are known to struggle with following the task as they tend to overfit to the scene. FiLM layer might be mitigating that helping the performance. 

3. Results on real world / more benchmarks.

VAT was only evaluated on one simulator. It would be nice to see if it retains performance across different benchmarks and ideally on real world data as well.

### Questions
Please see the weaknesses for my main concerns. My concerns are lack of ablations and weak motivation for the proposed method. I will be happy to update my scores if the authors can answer these questions and other reviewers do not bring up other major concerns. 

Other misc questions/suggestions:
1. Line 065: wrong? if dino is optimized for dense prediction tasks, then it would not throw away low-level local information?
2. In Fig. 3: it would be better to show the input task prompt as well
3. The quotation marks are wrong at many places. Maybe use `` to start quotes in latex.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
In this paper the authors propose VAT (Vision Action Transformer): starting from a ViT backbone, they insert “action tokens” at every ViT layer which are updated by an action module (i.e. a cross attention layer to the vision tokens). The main argument from the authors is that currently robot policies use only final layer ViT features and do not leverage the "representation trajectories" in all the ViT layers. The authors conducted experiments on LIBERO and reported 98.15% avg success, slightly beating baselines such us OpenVLA-OFT.
The authors also conducted ablation studies on skipping the final layers of ViT as well as using different camera views and losses.

Overall, the core idea is interesting, however the manuscript is in not a very good shape. I would urge the authors to heavily edit it, add significantly more experiments that validate their claims and describe in detail their protocols.

### Strengths
The idea behind action tokens cross attending on vision tokens in the full ViT hierarchy is nice.

### Weaknesses
1. Overall the main idea behind the paper is simple and offers very limited novelty. In many cases in the past features from various layers (not only for ViTs) can also be used for various tasks. The heatmap visualizations do not offer any additional insights and are kinda underwhelming as a support for the claim by the authors. The authors need significantly more experiments to validate their claims.

2. There is no established protocol regarding the experiments in Table 3. It is unclear how the other baselines are used (what views they use etc). There is also no explanation what the words scratch, fine-tuned, etc in the same table actually mean and what the authors did. Furthermore, important metrics regarding how the success metrics were calculated (such as number of eval episodes) are missing

3. The ablation studies do not seem interesting and importantly the losses and the views are not parts of the proposed model. Ablation studies usually need to validate every component of an architecture (e.g. here FiLM, action module parameters, action tokens, etc).

4. Overall the paper is poorly organized and written and results and tables are all over the place. I would suggest to the authors to better explain their work.

5. There is no baseline comparing features from the last layer of ViT directly for action. This is an important omission.

6. Claiming VAT as lightweight and then proceeding to train all layers of ViT is kinda non-intuitive.

7. Important: Using a Task ID embedding directly is kinda "cheating" -> it tells the model what policy to select before looking at the scene, and without relying on language. Could this could be a reason why the authors achieve high results comparable to other methods?

### Questions
1. Please explain the protocol for testing clearly.
2. How does a vanilla ViT (e.g. DiNOV2) features from the last layer fare against your method ? 
3. How important was FiLM ? 
4. If we dont include Task ID but use language, would your results still hold ?

### Soundness
1

### Presentation
2

### Contribution
1
