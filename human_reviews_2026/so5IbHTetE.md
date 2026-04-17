# Asymmetric Training with Heterogeneous Losses: A Probe into Architectural Resonance

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Is deep learning generalization necessarily rooted in optimizing a single objective? We explore an alternative view: adaptive generalization may emerge from structured interactions among heterogeneous objectives. We propose an Asymmetric Training Paradigm that temporarily introduces non-competitive, per-class supervision (Sigmoid losses) into networks optimized with competitive softmax objectives. This is realized through orthogonally initialized auxiliary pathways, modulated by a scalar coefficient $\alpha$ and present only during training. Crucially, we employ strictly controlled experiments to rule out parameter count as a confounder, identifying that simple parameter expansion yields zero gain. Our mechanistic analysis reveals two effects: (1) The proposed topology (but not mere capacity) consistently smooths the initial optimization landscape. (2) Final performance exhibits an architecture-dependent pattern we term Architectural Resonance, where auxiliary signals benefit models only when aligned with inductive biases. A 6-block Vision Transformer (ViT-6L) exhibits constructive gradient alignment (cosine similarity $+0.19$), yielding absolute accuracy gains of $+9.2\%$ on CIFAR-100. By contrast, a CNN shows destructive conflicts (cosine similarity $-0.26$). We further corroborate this divergence in hybrid architectures (CoAtNet), highlighting a stage-dependent nature: transformer stages benefit from heterogeneity, while convolutional stages show limited compatibility. We validate scalability on ImageNet-1k, showing consistent top-1 gains for ViTs (up to $+2.25\%$ on ViT-B/16). Rather than functioning as a universal regularizer, our probe reveals that heterogeneous signals selectively benefit architectures with weak inductive biases (e.g., Vision Transformers), exposing a critical dependence between architectural flexibility and 
objective compatibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper designs an asymmetric training paradigm that temporarily introduces noncompetitive class supervision (sigmoid loss) into a network optimized with a competitive softmax objective. This mechanism is implemented via orthogonally initialized auxiliary paths, which are controlled by a scalar coefficient α and activated only during training. This controllable form of temporary topological redundancy provides an ideal probe for studying target interactions.

### Strengths
The PPL is clear. The content is logical and there are no issues. The asymmetrical method is quite good in high level.

### Weaknesses
The study's core mechanism design, the "single activation" strategy, suffers from serious logical flaws and misattributes its role. The authors claim that the sole purpose of adding a large number of static auxiliary branches that are never activated or trained is to smooth the initial loss landscape. This claim is highly unusual and unconvincing. Attributing the observed stark differences in performance (ViT benefits, CNN suffers) solely to the geometric effects of this static topology on the initial landscape is a significant leap in logic. A more straightforward and likely explanation is that this peculiar structure introduces an unexplored implicit regularization, or that different architectures simply have vastly different sensitivities to the hyperparameters (particularly the value of α) of this structure and heterogeneous loss combination. The paper fails to provide strong evidence to rule out these simpler explanations, placing its core conclusion, "architectural resonance," on shaky ground.

The paper overgeneralizes the phenomena observed in a highly controlled and simplified experimental setting to a universal "architectural resonance principle," significantly undermining the persuasiveness of its conclusion. All experiments were conducted on small datasets (CIFAR-10/100) and specially designed lightweight models (e.g., ViT and CNN with only 6 layers). The inductive biases and optimization dynamics of these "toy models" are likely very different from those of deeper and more complex standard models (e.g., ResNet-50 or ViT-B) pre-trained on large-scale datasets like ImageNet. Promoting a regularity discovered in such a limited setting to a "fundamental principle" is both hasty and imprecise. Without validation on larger and more representative models, these findings should be viewed as an interesting preliminary case study rather than a generalizable principle that can guide practice.

The study's experimental design failed to effectively decouple and control variables, resulting in flawed causal arguments. The paper claims to follow a "practical gold standard" hyperparameter search strategy, namely, fixing the optimal learning rate and weight decay for the baseline model and then searching for the optimal auxiliary loss weight α only for the asymmetric model. This approach is methodologically problematic. Introducing a completely new loss term that potentially conflicts with the primary task significantly alters the optimization dynamics of the entire model, almost inevitably leading to changes in the optimal learning rate and weight decay. Therefore, the observed performance differences are likely not a pure manifestation of "architectural resonance," but rather the result of one architecture (such as a CNN) falling into severe suboptimal hyperparameter settings under the new training paradigm, while another architecture (ViT) happens to be more tolerant of this mismatch. This makes performance comparisons between different architectures unfair and weakens the persuasiveness of the conclusions drawn.

### Questions
The paper directly equates the phenomenon of negative gradient cosine similarity with a "destructive conflict" that is harmful to CNNs. However, in multi-task learning and regularization theory, conflict in gradient direction is sometimes viewed as a beneficial constraint, forcing the model to learn more general and robust feature representations that balance the needs of different tasks. Why do you a priori define this conflict as a purely negative effect, rather than a potentially beneficial regularization pressure that this particular CNN architecture cannot effectively exploit due to its capacity or inductive bias? How can we be sure that the observed performance degradation is the inevitable result of a failure of "architectural resonance" rather than a manifestation of an inability to converge to a better generalization point from this gradient "tug-of-war" due to insufficient model capacity, poor optimizer selection, or hyperparameter mismatch?

The principle of "architectural resonance" you propose is currently presented as a binary opposition: ViT is "synergistic" and CNN is "destructive." This division seems overly simplistic. Modern neural network architectures, such as ConvNeXt, have begun to combine the locality of CNNs with the global dependency modeling capabilities of ViT. According to your theory, how will these hybrid architectures respond to heterogeneous losses? Will they exhibit synergy, destructiveness, or something in between? More importantly, if this principle holds true, what specific implications does it have for future model design? Does it mean that we should completely avoid using non-competitive auxiliary losses in CNNs, or should we design new auxiliary signals that resonate with the CNN's local inductive biases?

The core of your experimental design—the "single activation" strategy—is quite unique. You argue that its purpose is to isolate the variable of "initial loss landscape geometry." However, this raises a key question: How do you prove that the full impact of these N-1 statically, orthogonally initialized branches on the training dynamics can truly be reduced to simply "smoothing the initial landscape"? Have you conducted the necessary control experiments to confirm this? For example, by introducing an equal number of static parameters connected in a different way (not in the auxiliary classification head) to verify whether the so-called "resonance" effect is unique to this specific topology? Without such verification, we cannot rule out that this phenomenon is simply a side effect of increasing the number of parameters, changing the weight initialization distribution, or introducing some unknown form of regularization.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies architecture-objective interactions by attaching training-only heterogeneous auxiliary branches (sigmoid BCE) to models whose main head uses softmax CE. Only one main branch is active during training; branches are removed at inference. The claim is a Principle of Architectural Resonance: ViTs benefit, CNNs degrade, MLPs mixed, explained via gradient cosine similarity between main and auxiliary losses. The paper was difficult to follow in several places, and some implementation details were not clear to me.

### Strengths
- Clear experimental knob: vary auxiliary-loss weight and number of branches, then measure gradient cosine similarity, not only accuracy.
- No inference overhead since auxiliary branches are removed.
- ViT gains on CIFAR-100 appear consistent with the reported alignment signal.

### Weaknesses
- Inactive branches vs. claimed smoothing: The paper states that only one branch is active in forward and backward passes, while the rest never participate. If so, it is unclear how additional inactive branches can smooth the initial loss landscape. This requires a precise computational explanation.

- External validity: Results are limited to small CIFAR models. No ImageNet-scale or strong pretrained backbones, so the generality of the claimed principle is uncertain.

- Overall, the paper was hard to understand, reducing confidence in the conclusions.

### Questions
- Do the inactive branches contribute in any way to the forward graph or loss computation? If not, how do they alter loss-landscape statistics? Please specify the exact computation path and any regularizers.
- Can the principle be validated on at least one ImageNet-scale experiment or a pretrained ViT-B and a stronger CNN with normalization and residuals?
- How does this compare against deep supervision baselines and gradient-conflict methods that explicitly manage alignment?

### Soundness
2

### Presentation
2

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
The paper proposes an Asymmetric Training Paradigm that adds temporary, orthogonally-initialized auxiliary branches trained with sigmoid/BCE losses to models whose main head uses softmax/CE. Only one auxiliary branch is active for backprop (“single-activation”); the rest remain fixed. The authors use this as a probe to study how heterogeneous objectives interact with architectural inductive biases, advancing the Principle of Architectural Resonance: auxiliary signals help when aligned with an architecture’s bias and hurt when misaligned. On CIFAR-100 with small backbones (MLP, 6-conv CNN, 6-block ViT), they report strong ViT gains (e.g., +25.4% relative over baseline at 20× redundancy), CNN degradation, and mixed MLP behavior. Mechanistic analysis includes gradient cosine similarity between main and auxiliary losses (positive for ViT, negative for CNN), training-dynamics comparisons, and attention-pattern measurements. The paper also explores “dose–response” curves over branch redundancy and some few-shot / label-noise settings.

### Strengths
1. Interesting question & framing. Treating auxiliary supervision as a heterogeneous objective and using it as a controlled “probe” is a fresh angle beyond standard deep supervision/MTL.

2. Mechanistic evidence, not just accuracy. The gradient alignment analysis (e.g., consistently positive for ViT, negative for CNN) gives a concrete signal-level explanation for the observed divergence and nicely ties to the “resonance” thesis.

3. Training-only overhead. The idea of introducing complexity only during training aligns with practical interest in keeping inference cost unchanged.

### Weaknesses
1. **Conceptual clarity of “single-activation” & landscape smoothing**. The paper argues that increasing static parallel branches (with only one active for backprop) “smooths the initial loss landscape” and influences optimization. It’s unclear why inactive, never-trained branches would alter the optimization geometry of the actual objective used for gradient updates if their gradients are blocked. If the total training loss truly reduces to $L_{main} + \alpha L_{aux}(active)$, why does the count of inactive branches matter for the landscape the optimizer “sees”? This feels under-justified and risks a confound in interpretation.
2. **Confounding factors (capacity & compute) during training**. Even if inference cost is unchanged, training adds parameters and forward compute. For large redundancy (e.g., 20$\times$, 300$\times$), the model class during training is materially different (Tables note parameter counts grow a lot). Gains might stem from implicit ensembling or extra linear projections rather than “heterogeneous objectives”. The paper needs tighter controls: 1. Parameter-matched training-only baselines (e.g., add equal-sized, randomly-initialized layers that do not produce auxiliary loss); 2. Capacity-matched trainable vs frozen auxiliary branches; 3. FLOPs/step and wall-time normalization across settings.
3. **External validity / scale**.Results are on small backbones and CIFAR-10/100. It’s hard to gauge impact without more complex setting like ImageNet-1k and standard architectures (ResNet-18/50, ViT-Ti/S). The claim that resonance is a principle would be much more convincing with one larger-scale confirmation.
4. **What exactly is “heterogeneity” driving?** The auxiliary head uses BCE per class. But would similar resonance show up if the auxiliary objective were homogeneous but perturbed (e.g., softmax CE with different temperature, label smoothing, or mixup-style targets)? Ablating the type of heterogeneity is essential to argue that “non-competitive vs competitive” is the key ingredient rather than “any extra supervision.”
5. **Clarity & rigor details.** 1. Please formalize how “inactive” branches enter the forward graph and loss computation. Are their losses computed but stop-gradded? Are their logits used anywhere else?; 2. Define precisely which loss surface is visualized for the “smoothing” analysis and why changes in inactive branches should affect it.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
2
