# Cooperative Hardware-Prompt Learning for Snapshot Compressive Imaging

- Decision: Reject
- Scores: 6, 8, 8, 6

## Abstract
Spectral snapshot compressive imaging (Spectral SCI) applies an optical encoder to compressively capture 2D measurements, followed by which the 3D hyperspectral data can be restored via training a deep reconstruction network. Existing reconstruction models are generally trained with a single well-calibrated hardware instance, making their performance vulnerable to hardware shifts and limited in adapting to multiple hardware configurations. To facilitate cross-hardware learning, previous efforts attempt to directly collect multi-hardware data and perform centralized training, which, however, is impractical due to severe user data privacy concerns and hardware heterogeneity across different platforms/institutions. In this study, we explicitly consider data privacy and heterogeneity in cooperatively optimizing spectral SCI systems by proposing a novel Federated Hardware-Prompt learning (FedHP) framework. Rather than mitigating the client drift by rectifying the gradients, which only takes effect on the learning manifold but fails to solve the heterogeneity rooted in the input data space, FedHP learns a hardware-conditioned prompter to align inconsistent data distribution across clients, serving as an indicator of the data inconsistency among different coded apertures. Extensive experiments demonstrate that the proposed FedHP coordinates the pre-trained model to multiple hardware configurations, outperforming prevalent FL frameworks for 0.35dB under challenging heterogeneous setting. Moreover, a new Snapshot Spectral Heterogeneous Dataset (SSHD) has been built upon multiple practical spectral SCI systems. We will release the data and code to enrich further exploration of this practical computational imaging problem.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on the robustness, efficiency, and accuracy of current snapshot compressed imaging reconstruction networks. The major contribution of the paper is designing a prompt network which automates the process of aligning a measurement based on its corresponding measurement model.

### Strengths
The paper tackles an interesting distribution shift, that is a shift in the measurement model of the compressed sensing task.

The paper is overall well-written (although in some parts difficult to read).

The idea of the prompt network to tackle distribution shifts is very interesting. If I understood correctly, without a prompt network, fine-tuning is needed for new measurement models. Yet with the prompt network the process of measurement alignment with the measurement model is automated for any measurement model.

The experiments are interesting and carefully designed in the sense that reasonable baselines and datasets are chosen for evaluation.

### Weaknesses
The major contribution of the paper is not well-justified. i.e., the crucial need for the proposed method (as opposed to training from the scratch for every new measurement model) is not well-supported. e.g., the reviewer still finds it very convenient to train a model for every new sets of measurement models for a new organization of interest from a practical perspective. i.e., all it takes is a few hours (days) of training for the new set (note that this automatically addresses the other concern raised by the authors regarding privacy constraints, in that each organization has access to its own data and device sets).

The comparisons are not fair in Fig. 1 (also please see my question regarding Fig. 1 below). Clearly, joint training should serve as an upper bound on the performance when the test set contains the same measurement models as training. 

The results (especially the quantitative ones) do not yield the conclusion that FedAVG is outperformed by FedHP. The major advantage of FedHP seems to be its 4x more efficient training time compared to FedAVG. This is fine and improving the efficiency is valuable from a practical point of view, but the paper isn’t oriented around this conclusion; the paper emphasizes the value of prompt networks and FedHP in the form of accuracy and robustness gains, whereas FedAVG enjoys those traits, too!

Minor:
On Tab. 2, FedHP is highlighted as the best-performing method in terms of SSIM (0.8481), whereas FedAVG should be highlighted (0.8496).

### Questions
How’s Fig. 1 obtained? Is it evaluated on the same measurement model set used during training of each setup? Or are all models evaluated on the same predefined test set of measurement models?

As mentioned in the strengths section, the idea of the prompt network is interesting. However, the biggest question raised is whether that network induces another source of instability to the overall model. Specifically, what guarantees that the prompt network doesn’t do a terrible alignment for measurement models deviating from the training distribution?

What is the source of inconsistency between Tab. 1 and Fig. 3? What we see in Fig. 3 flags for a higher PSNR difference than 0.14 dB between FedAVG and FedHP… It’s understandable to argue that quantitative metrics such as PSNR or SSIM don’t perfectly capture the true quality, but the visual difference on Fig. 3. is too large not to be captured by those metrics.

Why isn’t the deep unfolding network included in all the results and only reported as a short paragraph at the end? 

Any intuition on why FedAVG is so much slower to train than FedHP?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work develops a federated hardware-prompt learning (FedHP) method for the task of snapshot compressive imaging (SCI). Existing reconstruction methods generally consider a single well-calibrated hardware configuration for network learning, inducing a highly coupled relationship between the reconstruction model and hardware settings. Differently, this work adopts federated learning to coordinate multiple clients with variant hardware settings and proposes a hardware-oriented solution to mitigate heterogeneous data issues.

### Strengths
• The motivation of this work is impressive, pointing out a very practical problem for snapshot compressive imaging. Both the hardware cooperation and hardware heterogeneous problems are underexplored.  This work solves the heterogeneous issue accounting for the special characteristics of SCI.
• The design of the hardware prompter bridges the hardware and software in a novel way, which could be easily incorporated into optimizing diverse set-ups in SCI. 
• Experimental results are abundant and have shown a clear performance boost over previous methods. Extensive ablation studies and model discussions have also been provided.

### Weaknesses
• It remains unclear if the proposed method can adopt a larger client number. A detailed discussion on the number of clients should be given to demonstrate the practicality of the proposed method and to enhance the soundness of the work. 
• Is it possible to apply the proposed method to other hyperspectral image datasets? 
• It seems that a competitive method of FedGST for comparison was a centralized learning strategy, is it a fair comparison or what are the modifications toward this method? Please provide more details.

### Questions
• Is the dataset split of the centralized learning the same as the federated learning? Please provide more illustrations and details. 
• There are some typos in the manuscript, for example, Fig.3 caption.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper has studied a new problem for snapshot compressive imaging (SCI) by optimizing a cooperative network across different hardware configurations (coded apertures). A new hardware prompt learning module has been proposed and integrated into the FedAvg algorithm to enable co-optimizing multi-hardware and the global model for a computation imaging task. Extensive experimental results were provided on simulated and real data, compared with several federated baselines.

### Strengths
1）It is interesting and practical to leverage a federated learning framework to address hardware shifts across different systems while preserving the privacy of each system’s local data. Plus, the paper has collected data from multiple real hardware systems to empirically validate the proposed method. 
2）The proposed hardware prompt is a novel and efficient solution to mitigate data heterogeneity for developing deep SCI models in a federated learning framework, especially to enable co-optimizing multiple hardware and a global model across systems. A detailed ablation study has also been provided to clearly show the improvement given this prompt design. 
3） A multi-hardware dataset has been collected and built for this new problem, which could broadly benefit the SCI community. Extensive experimental results on multiple settings were provided in terms of both quantitive and qualitative evaluation.
4）Several state-of-the-art federated learning methods have been developed for a computational imaging task and been involved in the experiment comparison.

### Weaknesses
1） While federated learning is a good choice, it remains unclear if the proposed problem setting can be directly solved by some other simple solutions, such as meta learning or deep ensemble. 
2）Despite the large improvement given by the hardware prompt, it lacks further analysis of how this design works for different hardware. For example, will different hardware lead to different prompts? What these “hardware prompt” look like? Is the prompt network only implemented by an attention block?

### Questions
1）What are the benefits of introducing adaptors? Why not directly update the full model?
2）What’s the main reason for setting C=3 in the experiment? 
3）In Eq (9), is there any other way to impose a prompt on the measurements? For example, can the concatenation operation be applied?
4）It would be better to directly explain the settings of different hardware shits in the captions of Table ½.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Motivated by recent success of Federated Learning (FL) and Prompt Tuning, this paper proposes a deep neural network framework, named as FedHP, that can take into account diverse sensor acquisitions for spectral snapshot compressive imaging (Spectral SCI). The primary distinction from existing FL methods lies in the inclusion of a measurement enhancement network that considers both degraded observations and the physical forward model pattern across clients. Experimental results demonstrate its effectiveness of FedHP on both simulation dataset and real-world SCI dataset.

### Strengths
1), This paper is overall well written and easy to flow. It clearly introduces the motivation and problem formulation, making the method accessible to non-SCI experts.

2), Both simulation and real-world datasets are considered, making a better practical contribution.

3), The experimental comparison is comprehensive, and baseline methods are up to date.

### Weaknesses
1), The technical contribution to more general computational imaging seems to be limited or at least not well supported by this paper’s current state.

2), Likewise, the main deep learning technic behind this proposal, FedAvg, is already well known, which makes the technical contribution to deep learning community also marginal.

3), The idea of using another learning-based module that can consider additional forward-model settings seems not new to model-based deep learning methods for computational imaging. Moreover, it is difficult to evaluate the proposed “correction” module indeed robust to distribution shift. At least, there is no clear evidence presented in this paper.

### Questions
1), Figure 1. It is difficult to find differences between 4. FedAvg and 5. FedHP, the method instruction plot.

2), The authors did not discuss a lot about why their method robust to the codec pattern shift, both intuitively and theoretically. What if the new module $\phi$ cannot handle very new coded aperture $\bf M$?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
