# Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Model Reprogramming

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 2, 8, 2

## Abstract
Black-box model reprogramming (BMR) aims to re-purpose black-box pre-trained models (i.e., APIs) for target tasks by learning input patterns (e.g., using Zeroth-Order Optimization (ZOO)) that transform model outputs to match target labels. However, ZOO-based BMR is *inefficient*, requiring *extensive API calls* (could be expensive) and suffering from unstable optimization. More critically, we find this paradigm is becoming ineffective on modern, real-world APIs (e.g., GPT-4o), which can ignore the input perturbations ZOO relies on, leading to negligible performance gains. To address these limitations, we propose **PoRL** (Prime Once, then Reprogram Locally), an alternative strategy that shifts the adaptation task to an amenable local model. PoRL initiates a one-time priming step to transfer knowledge from the service API to a local pre-trained encoder. This single, efficient interaction is then followed by a highly effective white-box model reprogramming directly on the local model. Consequently, all subsequent adaptation and inference rely solely on this local model, *eliminating* further API costs. Experiments demonstrate PoRL's effectiveness where prior methods fail: on GPT-4o, PoRL achieves a +27.8\% gain over the zero-shot baseline, a task where ZOO provides no improvement. Broadly, across ten diverse datasets, PoRL outperforms state-of-the-art methods with an average accuracy gain of +2.5\% for VLMs and +15.6\% for VMs, while reducing API calls by over 99.99\%. PoRL thus offers a robust and highly efficient solution for adapting modern black-box models.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper claims to present a method for black-box model reprogramming to adapt large model outputs to solve a downstream task. The method, however, boils down to training of a small network to solve the downstream task, which is initialized before training by a (some?) query (or queries?) of a large black-box model. It does not actually do any black-box model reprogramming or adaptation of the outputs of a black-box model.

### Strengths
The proposed approach requires orders-of-magnitude fewer API calls during training and inference, yet still outperforms other approaches (e.g ZOO methods), which don't even improve upon zero-shot performance. (but see the weakness section where I argue that this is misleading).

### Weaknesses
It is not clear what the blue flower block in figure 2 is supposed to represent. Is this a visual prompt? The caption should be more informative.

It is not clear where the "locally available pre-trained encoder" comes from or what its characteristics should be. What purpose is the local encoder supposed to serve here? Is it just computing useful features? If so, how does one know what would be useful for a given downstream task? 

As written, the approach doesn't make sense, and appears to boil down to solving the downstream task with a simple network.
On line 084: "we query the service API just once", but on line 091 we hear that the number of API calls is 10^3. Why the discrepancy?  
If the service API is only called during the priming step, then doesn't the method boil down to just training of a relatively small model (the local pre-trained encoder plus the lightweight linear layer) to solve the downstream task? All the enormous power of the service model is  only being used as a priming or initialization step for the lightweight network.
I may be missing something here, but the proposed approach seems to be just a very simple technique, where the only thing new is a method of initializing (or as the paper mentions it is similar to a distillation) the network from a large service API.

The so-called "cost-free inference" is very misleading. Of course it is cost-free - you don't use the large service model at all during inference, just the small network. It is as though we used GPT to initialize or distill AlexNet for a few iterations and then claim that we are somehow doing "black-box model reprogramming" (as the paper title suggests).

### Questions
See the weakness section for questions to answer.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Existing zeroth-order optimization approaches for adapting API-based black box models to downstream tasks are highly inefficient as they require extensive API calls. Also, robustness of modern API-based models to input perturbations makes zeroth-order optimization approaches ineffective. To address these issues, this paper proposes to simply distill the API-model to a local (pretrained encoder + linear classifier) model and then adapt the local model to the target task. In summary, the proposed approach advocates distillation followed by further adaptation.

Experiments were conducted using 10 downstream datasets with multiple teacher-student combinations and the proposed approach is shown to outperform approaches that perform black-box adaptation directly on the teacher API model.

### Strengths
Overall the paper is written well and was easy to follow.

Focuses on the practical problem of cost-effective adaptation of black-box API models

### Weaknesses
**Lack of novelty**
* Distillation is a decade old concept in ML community and distilling the predictions of a frozen teacher model to a student model is widely used. This paper is simply repackaging "distillation followed by further adaptation to target tasks" as API model adaptation.

* While the paper takes about real-world API models in the introduction to motivate the problem, the actual experiments are done using standard CLIP models and vision models. Basically, the paper is running experiments in few-shot distillation settings with standard CV models and datasets and is rebranding it. There are numerous SoTA open sourced CLIP and Vision encoder models out there today and distilling from those models would eventually lead to much better performance on the target datasets used in this paper.


**Few things in Fig.1 needs to be clarified**
* In Fig 1 (b), does #API calls include both training and inference API calls? 
* Does x-axis in Fig. 1(a) represent training cost or inference cost or both? 
* Which dataset is used for creating these plots?

### Questions
See the weaknesses mentioned above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The proposed method aims to improve query efficiency of black-box adaptation approaches. Instead of repeatedly querying the API inference service for optimization, the authors leverage a locally available pretrained encoder and perform a single “priming” query with downstream task data. The resulting outputs are used to train a lightweight linear layer that aligns the local encoder’s representation space with that of the black-box model. This one-time transfer eliminates the need for further API calls, substantially reducing adaptation cost while preserving task-specific performance. The proposed one-shot priming resembles a micro-distillation step, transferring black-box model knowledge to a local encoder while minimizing API queries.

### Strengths
* The paper presents a clear and well-motivated objective, optimizing API-based inference efficiency. The authors empirically identify the limitations of existing approaches, providing a sound scientific rationale for their proposed technique.
* The manuscript clearly introduces the key terminologies and foundational concepts necessary to understand the contribution, ensuring accessibility for readers from related domains.
* The contribution is well-situated within the existing literature; a comparative table effectively highlights how the proposed method differs from and improves upon prior work.

### Weaknesses
* While the overall presentation of the paper is strong, certain sections would benefit from additional proofreading and stylistic polishing, particularly the abstract, to improve clarity and flow.
* The selection of inference services and local models evaluated appears somewhat limited. Given that the main contribution focuses on reducing API inference calls, incorporating even a small-scale evaluation using an actual proprietary API service (e.g., OpenAI’s CLIP-like endpoint or Gemini Vision) would strengthen the empirical validation and make the results more realistic and convincing.

### Questions
* How do you anticipate companies offering proprietary inference services will perceive the adoption of your technique? Do you foresee any potential updates to their usage policies as a response?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method to tackle black-box model reprogramming, which is the problem of learning a (visual) prompt to applied to input images to achieve better results on a frozen black-box model (e.g. commercial models deployed behind APIs which only allow input-output interactions). Unlike existing methods, the paper proposes a method that does not rely on Zeroth-Order Optimization (where gradients need to be estimated via changes in output losses given small perturbations of inputs), which is typically costly in terms of API calls and training time. This is done by using a local model to approximate the behavior of the black-box "service model". To accomplish this, a set of model outputs or logits are first generated by querying the local model with the training set. A linear layer is then trained on top of a pre-trained local model to adapt it to behave similarly to the service model. The local model is then prompt-tuned, and used for subsequent inference without relying on the service model. Compared to methods based on zeroth-order optimization, the paper shows that the proposed method achieves state-of-the-art performance across multiple benchmarks, while API calls are reduced by over 99%.

### Strengths
- The paper offers an efficient alternative to zeroth-order optimization for test-time adaptation of black-box models. This is an important and relevant problem, given the high number of API calls (shown to be on the orders of $10^8$) and long training time (shown to be over 32 hours). This makes such methods significantly more practical, and less costly, to be deployed in practice.

- The method is also shown to achieve state-of-the-art performance across a large variety of datasets.

- I think that the ability to leverage unlabeled data to achieve better alignment of the local and service model to further improve results is a strong point, since the alignment / "priming" stage simply requires distilled pseudo-labels from the service model.

### Weaknesses
- Upon initial review of the paper, I would expect that the method learns a visual prompt with the local model, which is then used for further inference with the *service* model. Instead, it was surprising to see that the paper proposes to discard the service model altogether, and use in its place the local fine-tuned model. This appears to defeat the entire purpose of having a powerful service model -- if one can easily distill a local model that performs better than a large commercial service model, then what is the point of having a service model in the first place?

- The paper seems to be missing a key data point in all of its experiments, which is necessary to assess the value of the proposed method:  What if one simply tunes the local model (full fine-tuning or linear layer) with the new dataset, how does it compare to using the service model for "priming" the local model?

- A use case of service models, black-box or not, is that they can be accessible for those without the necessary compute to run local models. As such training a visual prompt that can subsequently be added to all input images is attractive, since it only requires a one-time training cost, and subsequent users of the model can simply take the resulting visual prompt and append it to their queries. However the proposed method here does not train a visual prompt that works well for the service model, and instead requires all subsequent inference to be done on a local machine with the necessary compute resources to run it. This also seems to defeat the purpose of reprogramming black-box models in the first place, since it simply replaces the service black-box model with a local white-box one.

- The theoretical section appears rather self-fulfilling. In particular, to prove that the risk of the local and service model are close, the paper requires 
> Assumption 2. (ϵ-Faithful Priming). The priming process is effective, resulting in the local model $F_L$ closely mimicking the logit distributions of the service model $F_S$, both immediately after priming (before VR) and after both models have been optimally reprogrammed for the downstream task.

This assumption is extremely strong, since it essentially assumes the desired conclusion that $\epsilon$ in eqn (3) is small. What should be proven and not assumed is that the "priming process is effective" for certain classes of models, under reasonable conditions. Consequently, the theory simply proves that the risk difference between the local and service model is bounded by some $\epsilon$, but says nothing about how small or large $\epsilon$ is, which makes it less meaningful.

- For the VLM experiments, the local and service models seem to be extremely similar architecture wise -- e.g. in VLM experiments, they paper uses "CLIP ViT-B/16 as the service model and ViT-B/16 as the local encoder". In the VM experiments, the service models are also architecturally very similar to the local ones (e.g. ViT-B/16 vs ViT-B/32, ResNet101 vs ResNet50), I assume likely trained on very similar, if not the exact same, datasets as well. It is not clear that this method will work on the scale of commercial API models as advertised, since those that can run locally are likely to be vastly different, in terms of both architecture and training recipe, from that offered commercially. It seems unlikely that one can be reconstructed from the other via a simple linear transformation, else this implies that the commercial model itself is not that valuable in the first place. 

- I am aware of some discussion in Appendix D.8. regarding this, but do not see any details about the local model used, nor comparison tables containing results and performance of other methods. 

- PoRL-MS which falls back upon expensive API zero-shot inference feels like a patch that undermines the paper's core premise, which is to achieve cost-free inference with a local model. 

- As the authors mentioned, the method does not work well in extreme few shot scenarios.

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1
