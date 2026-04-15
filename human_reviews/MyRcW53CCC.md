# PrivacyRestore: Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restoration

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 5

## Abstract
The widespread usage of online Large Language Models (LLMs) inference services has raised significant privacy concerns about the potential exposure of private information in user inputs to malicious eavesdroppers. Existing privacy protection methods for LLMs suffer from either insufficient privacy protection, performance degradation, or large inference time overhead. To address these limitations, we propose PrivacyRestore, a plug-and-play method to protect the privacy of user inputs during LLM inference. The server first trains restoration vectors for each privacy span and then release to clients. Privacy span is defined as a contiguous sequence of tokens within a text that contain private information. The client then aggregate restoration vectors of all privacy spans in the input into a single meta restoration vector which is later sent to the server side along with the input without privacy spans.The private information is restored via activation steering during inference. Furthermore, we prove that PrivacyRestore inherently prevents the linear growth of the privacy budget.We create three datasets, covering medical and legal domains, to evaluate the effectiveness of privacy preserving methods. The experimental results show that PrivacyRestore effectively protects private information and maintain acceptable levels of performance and inference overhead.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work operates in a setting where the user wants to off-load inference computation to a server while preserve the privacy of its input data. The authors propose a framework that trains restoration vectors during the preparation stage, then have the clients transmit meta vectors with masked inputs to the server, which will be used for inference by the server.

### Strengths
1. The paper focuses on an important problem of privacy-preserving inference of large language models
2. I believe having a hybrid privacy setting where the input is a mixture of private and public information is more practical. In particular, the idea of a privacy span is interesting.

### Weaknesses
1. There are a lot of moving parts in the framework that make it difficult to understand and follow. Rather than only elaborating on the technical aspects of each step, the authors should also discuss a running example, similar to the example user input used in Figure 1, to help give more intuition about what each step is doing. 
2. It would be beneficial if the authors could provide a background section on the attention mechanism and activation steering methods prior to introducing the methodology. 
3. In the preparation stage, the server supposedly has access to user inputs in the training set. But during inference, the framework is providing privacy to another set of user inputs? So the authors are not concerned about the privacy of this user input training set? 
4. The assumptions needed for the textual data, which are stated in the introduction, are a bit strong. For example, in Figure 1, the user input in the training set is nearly lexically identical compared to the inference user inputs. So, do we also have to assume that the server has access to this training set? All of these assumptions may severely limit the scope of the work to a small set of datasets. 
5. Additionally, the client needs a local lightweight model for the inference stage. This feels like it defeats the motivation of this work: users offload ML computations to a server.
6. Due to weaknesses (3), (4), and (5), I’m unsure what real-world application this framework falls under. Especially with (5), if a client needs to perform ML computations, why would one use this framework over federated learning? The client can just run inference locally with their lightweight model trained via Federated Learning + client-level DP-SGD, which achieves LDP–a stronger version of dx-privacy.

### Questions
See weakness (2), (4), and (6) for questions.

### Soundness
2

### Presentation
2

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
The paper proposes a method ('PrivacyRestore') to protect private information in user inputs that are sent to a cloud LLM service for generating responses during inference.

As preparation, their solution first identifies task-specific contiguous sections, so-called "privacy spans", that represent privacy-sensitive information.
Next, for each privacy span, the attention heads in the LLM are identified which are most affected when that privacy span is removed (redacted) from the input texts. Based on this, a subset of the common top K most influenced attention heads is selected across all identified privacy spans.
Now the server trains so-called "restoration vectors" for each privacy span and each of the top K attention heads, which can be used to correct the output of the attentions head when the privacy span is redacted in the input text.
The restoration vectors are then sent to all clients.

During inference, if a client wants to use the LLM service, it first determines all privacy spans in its input text and computes a normalized, weighted sum of the corresponding restoration vectors for each of the top K attention heads. The weights are determined using a separate, light-weight proxy model (BERT) that is run locally on the input text. For additional privacy protection, the summed vectors are perturbed with Gaussian noise to achieve d_X-privacy, a generalization of differential privacy. 
The client the redacts the privacy spans from the input text and sends the redacted text together with the noisy, so-called meta vectors, to the server.
The server runs the redacted input text through the LLM while using the meta vectors to correct the outputs of the affected attention heads, and sends the generated text back to the user.

The authors derive three datasets for conducting privacy-relevant evaluations. They propose a variety of experiments involving suitable baselines and interesting attacks.
However, some theoretical issues and questions remain that in my opinion must be clarified, which is why the current results should be taken with a grain of salt.

### Strengths
Comparisons against d_X-privacy token-perturbation methods on the entire input or only privacy spans, and against paraphrasing, provides interesting baselines in the experiments.

I also like the proposed attacks for the experiments on empirical privacy protection. The proposed prompt injection attack to output the original text is a good additional idea to check how well sensitive information is actually protected. On the other hand, it seems the inversion attacks could be strengthened (cf. weaknesses).

Overall, I think the approach could work in practice, albeit with the limitation of how well sensitive information can be covered through a limited set of privacy spans that must be defined in advance. (cf. weaknesses: typos in names of diseases, etc.).
More importantly, given the issues in the Theorems 5.1 and particularly 5.2, I am not confident that the experiment results use the correct privacy budget for the proposed method.

### Weaknesses
Theorem 5.1:

- What is the context/what assumptions do you make how DP methods are utilized here? The way you formulate Theorem 5.1, it reads as if it applies to any LDP/CDP/d_X privacy mechanism. However, that makes no sense since DP can be applied in other use cases apart from protecting text, and even for text, it depends on how the DP mechanism is applied to a text to make conclusions about the required privacy budget. So you should specify the actual scope to which your Theorem 5.1 applies to.


Theorem 5.2:

- Since you transmit meta vectors R_h=Z_h+noise for each affected/'edited' attention head, shouldn't the privacy budget also scale with the number of affected attention heads |H_c| ? Note that this also would affect your experiment results: You state in L397-398 that there are 175 or 125 affected attention heads, which would significantly increase your total required privacy budget!
- Can you provide an upper bound for the ||Z_h'-Z_h||? If they are normalized unit vectors and the norm is the L2 norm, 2 might work?

L71-L73: What happens if there is a typo in name of disease or symptom? What if new symptoms are discovered for a certain disease? What if a disease or symptom is paraphrased in an unexpected way? It seems, such examples cannot easily be covered by privacy spans and hence often remain unprotected, so the assumption about limited number of required privacy spans appears overly optimistic to me.

Restoration vectors are determined on target LLM (Llama2) per attention head, but the associated weights to combine these restoration vectors into the meta vector are determined from another proxy model (BERT) with _different_ attention heads.
How does this fit together (e.g., due to possible deviations in these models' attention scores), especially since the weights are averages across the proxy model's attention heads, but then still applied individually to the per-head restoration vectors?

### Questions
Please introduce or explain special terms and concepts *before* using them:
- L019: What is a "privacy span"? (Yes, it is in the next sentence, but these tiny details make the paper hard to read.) Can you reorder/rephrase to describe this concept first?
- L025: You talk about "linear growth of the privacy budget", which seems to imply that some form of differential privacy is employed. However, you never mentioned differential privacy as part of your solution until that point.
- L213: What is "*the* Top-K Heads Selector"? This only becomes clearer after reading the paragraph. Instead, maybe start the section explaining that you subsequently "will introduce *a* Top-K Heads Selector whose aim is to ...".
- L269: "\lambda represents *the* coefficient" <- Which coefficient? What does it do? Where did you introduce it?

L068-L070: What do you want to say here? "if privacy spans [...] are removed [...] attackers are able to recover private information" *despite* having removed the privacy spans? Or is this an (intentionally bad) example where the spans are too short (covering one word only)? In this case, it may help to explicitly distinguish proper vs. degenerate spans (longer contiguous sequences vs. single words).

L205-210: I find the description of 'probes' a bit too complex. Do we need the formula and knowledge of its internals like the parameters? Or could this be stated more simply as "a probe is a binary classifier consisting of a dense layer with a sigmoid activation" (or similar)?

L253: (typo) "server" side

L269: In "a is the initial output give[n?] the complete input", what output do you mean? Output of some attention head, some other part of the LLM, the entire LLM (softmax probabilities over the token space)?

L252-272: (Related to the previous question.) Where and how would the restoration vectors be applied to actually restore something (if no meta vector was produced)? You just described how to compute/train them, but not how and where they would be applied.

L294 (and elsewhere): You talk about "meta vector R_h on head h", so there is not just one meta vector, but several meta vectors, one for each affected/'edited' attention head h \in H_c? Then I would assume that not just one, but all of these meta vectors are computed by the client and sent to the server. Elsewhere (e.g., L241, L277, and others), you often refer to 'the' or 'a' meta vector in singular form. Note that this is important also for determining the overall privacy budget, cf. weakness on Theorem 5.2.

L345-355: You state the number of "_types_ of privacy spans".
- What is the meaning of a 'type'? Can you give one or two examples?
- How many actual privacy spans (not 'types') did you identify for each dataset?
- What is the impact of the number of spans on the computational overhead? As I understand the approach, the overhead grows with the number of privacy spans, since inference requires clients to compute attention weights by running a separate proxy model (BERT) for each possible privacy span.

Minor question: Why do you name the affected attention heads "edited" instead of just "affected attention heads"? As I understand the paper, the heads that are most affected by the removal of the privacy spans are left unmodified (i.e., *no* editing is done to them), and by applying the meta vector you only modify the _output_ of those attention heads. (Besides, you already have another name for them, e.g., you write "common top-K heads set H_c" in L316.)

It may help to expand preliminaries (could add into related work section) on concepts that are critical to the approach proposed in the paper, to make the paper more self-contained and to make it easier for readers that do not yet know the cited papers where these preliminary concepts have been introduced:
- Add preliminaries on _inference steering_, including restoration vectors.
- Might help to recap attention mechanism/heads (since they are a critical component used for the probes and activation steering).
- Alternative: Could split "Related Work" (what other approaches that you are *not* building upon exist, such as SMPC, but also d_X privacy perturbations or paraphrasing that you use as baseline) and add separate "Preliminaries" (background on building blocks that are critical for your solution, like DP, inference steering, attention mechanism details) section. (I am aware of space constraints; one idea is that you could shorten elsewhere, e.g., you write rather frequently, in bold, what privacy spans are.)

Section 6.3:
- You use an embedding inversion attack, but I didn't fully understand how exactly you implement it (even after checking Appendix L). My guess is: The attacker tries (using a finetuned GPT-2) to predict the original text (i.e., the privacy spans) given the meta vectors? But then a stronger attack model (that I still think is plausible) could provide the attacker some context around the privacy spans, e.g., the incomplete queries, in addition to the meta vectors. Then the attacker could train a model to predict the privacy span given the context (infilling/masked language modeling) *and* the noisy meta vectors, which could narrow down the predictions to more suitable ones.
- In the "different d_X-privacy Percentage" experiments, you vary the percentage of protected tokens, but I'd assume that you leave the overall privacy budget the same? For clarity, I suggest stating this assumption explicitly.
- The blue curve (Pri-NLICE) in Fig 3.(c) seems to go up linearly and almost double -- which seems to contradict your expectations. Could you explain the observed behavior, and how it deviates from or still is in line with your expectations? Could more protected privacy spans actually lead to more information leakage, as more private information is encoded in the meta vectors that are used for the inversion attack?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a method for a privacy-preserving inference in LLMs, satisfying $d_\mathcal{X}$ privacy, a notion originally proposed as an adaptation of local DP to location data. Authors define *privacy spans* - a portions of text containing sensitive information. They then define a method to obtain a prediction from an API-accessed LLM, while sending a query with stripped privacy spans, plus private information encoded in a *restoration vector*. The server will have trained restoration vectors for existing privacy spans (of which authors assume a limited number of), to later use restoration vectors for activation steering. The authors created 3 privacy-focused datasets and presents experiments showing their method to outperform naive application of $d_\mathcal{X}$ privacy, as well as paraphrasing, for a task at hand.

### Strengths
The paper raises an important question of the appropriate privacy unit for text processing. While DP for language models has been studied extensively, the field lacks consensus on what threat model should be applied to text, and whether sequence-level or document-level approaches appropriately mitigate privacy risks.

### Weaknesses
## List of privacy spans.
The paper assumes a closed list of available privacy spans - less than 200 words/phrases used in the experimental settings. First, this introduces a privacy leakage from the mere fact of removing privacy spans - it reveals to the attacker that an original text contained one of the few "sensitive" items. Second, privacy is contextual, and defining an exhaustive list of sensitive phrases for a real-world use might be tricky, even if limited to a narrow domain, e.g. medical. What is considered safe for some people, will be sensitive for others.

## Privacy guarantees
I did not fully understand the threat model and the privacy guarantees provided. Throughout the paper, authors refer to attackers *intercepting* user queries (lines 41, 168), which suggests they are not concerned with protecting the sensitive queries from model developers. If that's the case, however, the standard encryption protocols should suffice to protect the traffic in-flight. Further, if an LLM developer is not considered a threat, an adversary should be considered that has access to the released restoration vectors and can perform the same computations as the server in the proposed setup.

Moreover, I did not follow how privacy guarantees are comparable across methods. On line 404 authors claim to ensure that all privacy methods are under the same privacy budget. Table 5, however, shows $\epsilon=75$ for PrivacyRestore, and significantly lower values for other methods.

In Theorem 5.2 authors claim to provide $d_\mathcal{X}$-privacy with a privacy budget depending on the distance between two restoration vectors. While I'm not very familiar with $d_\mathcal{X}$-privacy, typically data-dependent privacy budget merits a very careful proof, as the privacy budget itself could potentially leak private information.

### Questions
* The paper introduces SMPC in the "Related Works" section. How is SMPC relevant for the technique proposed in the paper?
* The metrics MC1 and MC2 (lines 1167-1176) used in the paper seem to simply refer to accuracy and entropy. The cited paper Lin et al., 2021 does not seem to define these metrics either.
* If MC1 is indeed an accuracy of selecting one out of 4 options, how is the performance of some methods drops consistently and statistically significant below 25% random guess baseline?
* In Figure 2, the attack success rate does not seem to grow significantly with increasing privacy budget. Do you have an explanation for that?
* In table 1, I would like to see non-DP baselines. First, a method that does not provide any privacy guarantees. And second, a method that simply removes privacy spans and asks LLM to provide an output based on the incomplete information. That would help put the performance of the methods considered into a proper context.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes PrivacyRestore to protect the privacy of user inputs during LLM inference. The user sends text with privacy spans removed along with the aggregated restoration vector to the server. The experimental results demonstrate that PrivacyRestore achieve a good balance between utility and privacy.

### Strengths
- The paper proposes a plug-and-play method.
- The authors conduct extensive experiments to validate the effectiveness of their methods.

### Weaknesses
- **Identification of privacy spans**. The authors didn't discuss in detail how to identify the privacy span in their paper. As is observed in the dataset they provided, there's still sensitive information contained in the input after the privacy span removed. For example, for the raw query "A 31-year-old male has a history of antipsychotic medication usage, nausea, stimulant drug use. The 31-year-old male presents the symptoms of involuntary eye movement, jaw pain, muscle spasms, ptosis, shortness of breath. What is the likely diagnosis?", the mask input is "A 31-year-old male has a history of nausea. The 31-year-old male presents the symptoms of muscle spasms. What is the likely diagnosis?". We can see that some key information, such as the age, gender, and symptom of the person, is directly reveal to the server. From this perspectively, the protection level of this method is limited, compared with applying DP or SMPC on the full sentence.

- **Reliability of restoration vector**. According to my understanding, the restoration vector for each head is the same for all privacy spans. Then how can the restoration vector help the server obtain better prediction for each specific privacy span? Even if the weights on each head are varied, the usability of the method is not persuasive enough. Furthermore, in the ablation study, the authors are recommended to compare the utility of PrivacyRestore, and the case without restoration vector.

- **Unavoidable privacy leakage from response**. As the framework directly utilizes the response generated by the LLM, it's inevitable that the server could infer the user's information from the plaintext response. Specifically, a high accurate response may contain sensitive attributes of the user, leading to a inherent trade-off between utility and privacy.

### Questions
- How do the users determine the privacy span in reality? What if privacy information spans the majority of the prompt?

### Soundness
3

### Presentation
3

### Contribution
2
