# RecSpy: Cognition-Driven PIN Inference on Randomized Soft Keyboards

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
As mobile devices have become deeply integrated into daily life, users often input sensitive data (i.e., PINs) to unlock services or authorize payments, which introduces high risks of side-channel attacks. To defend against potential attacks, in practice, soft keyboards for PIN entry are randomized in layout to mitigate such threats. In this paper, we present RecSpy, a novel cognition-driven acoustic side-channel attack that infers PINs on randomized soft keyboards. Unlike prior work that relies on video, power, or electromagnetic emanations, RecSpy exploits a previously unexplored vulnerability: distinct human recognition latencies for symbolic numbers. By modeling cognitive latency patterns and leveraging acoustic keystroke signatures, RecSpy learns individual and digit-level recognition features through contrastive and self-supervised learning. Furthermore, we also introduce a novel Logic-Guided Inference Network that integrates recognition patterns with the reasoning capabilities of a large language model (LLM) to prune the hypothesis space and infer complete PIN sequences. We extensively evaluate RecSpy on both Android and iOS devices, and results show that it improves the probability of successful inference by up to 4000×, which demonstrates a practical threat to current mobile authentication systems and shows that representation learning and LLMs can enable new side-channel attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Authors present RecSpy, a novel cognition-driven acoustic side-channel attack that infers PINs on randomized soft keyboards. RecSpy exploits distinct human recognition latencies for symbolic numbers. They model cognitive latency patterns and leverage acoustic keystroke signatures using contrastive and self-supervised learning to learn individual and digit-level recognition features. The authors also introduce a novel Logic-Guided Inference Network that integrates recognition patterns with the reasoning capabilities of a LLM to prune the hypothesis space and infer complete PIN sequences. The method is tested against Android and iOS devices. The method improves the probability of a successful PIN inference by up to 4000x compared to random guessing.

### Strengths
- **Novel focus on randomized soft keyboard for PIN entry**, which is commonly used as a countermeasure against fixed-layout attacks in practice. This makes the PIN inference problem harder than that on fixed-layout but more realistic and vital to investigate.
- **Novel concept of exploiting human cognitive recognition latency** for inferring PIN digits. This is novel and very interesting concept that also seems to be supported by referenced work. This approach enables a realistic threat model that only assumes microphone access through an installed malicious app on the victim's phone.
- The attack seems to generalize between different smartphone and OS platforms (Android and iOS).

### Weaknesses
- **The attack does not seem to be very successful.** Hit rates of 8%, 13%, and 29% for Top-20, Top-50 and Top-100, respectively, do not seem all that successful. Moreover, PIN entry usually has limits of 3 to at most 5 tries before being blocked by the apps (bank account apps or similar), making this very unrealistic.
- **Lack of justification and evidence-supported rationale for the steps and components of the proposed method.** While the paper includes includes an ablation study for the third part using LLMs, the rest of the process offers only vague or no reasoning for design choices. Each step of the method is packed with many different components (e.g., SENs, CNNs, LSTMs in just one part of the method). The selection of these specific components appears arbitrary, as the authors provide little to no justification or empirical support for their choices. This raises significant doubt regarding the necessity for such highly complex method, and makes the method challenging to evaluate and validate.
- **The provided explanation of the method is incomplete,** leaving some open questions about its overall process. Specifically, there is ambiguity surrounding the collection and application of user (victim) data. The authors fail to clarify when this data is collected in the process and how it is integrated and utilized by the method. The training schemes (paragraphs) require a more detailed discussion on what data is used and the precise data collection methodology. Moreover, a significant ambiguity exists in the method's ability to differentiate keystrokes: it is not clearly explained how the system distinguishes between inputs related to PINs and general smartphone usage, such as typing messages. 
- **The evaluation of the proposed attack method is significantly limited** due to its comparison against only random guessing. This minimal baseline restricts the proper assessment of the attack's true performance and efficacy.

### Questions
These questions are, in most part, addition to the stated weakness.

- My impression is that the results are not very good. Thus, what are the reasons for such a performance, and what makes these metrics and these results indicate a successful attack and a real threat to user's PIN security? Top-20 does not seem to be good enough for limited PIN tries, and with the statement from the paper, "an attacker can continuously collect data and attempt guesses over time", the question raises as to how long does it then take, and how much data is necessary?
- What are the implications of this work? Should smartphone designers/app designers do something and should users of those smartphone be concerned? This is also connected with the previous point on the attack's success.
- The terms utilized in this work, such as Individual-level Recognition, and Digit-level Discrimination, individual-specific patterns and digit-level latencies are not very clear. Moreover, they are never explicitly explained. For example, only in training scheme for IRM, it is said that the data is labelled with the age groups, meaning that this is where the reader learns that the individual-level recognition means the age group recognition specifically (if I understand that correctly). So, my suggestion is that these terms are explained more explicitly to avoid confusion.
- Why are all three steps of the method necessary? How do you select all the components and their combinations for each of the steps? Also, some of the hyperparameters are mentioned, such as number of epoch, and the choice for those is also not discussed.
- Another confusion stems from the data and data handling. 
	- There is a dataset in Section 3.3. (143 participants) and a dataset from Section 7.1 (40 participants). Is the 40 participants data a a subset of 143 participants data or a completely new group of participants? With the same group of people, there can be bias in the ML methods utilized.
	- Is it an assumption that the attacker has a similar dataset that is collected offline and labelled with age groups?
	- At which point is the victim's data collected (for inference) and how is it utilized in the method?
	- The data described in Section 3.3 is preprocessed by removing silence and aligned to a specific time. Is the collected data from victim user also preprocessed in this way?
	- From the collected victim's data, how does the attacker differ between keystrokes related to the PIN entries and the rest of the smartphone usage?
- Why was the attack method only compared with random guessing?

Additional comments/questions (less relevant)
- How does your approach work if the keyboard layout is not randomized? 
- "but the relative differences among digits remained unchanged." - This is not true, look at times for numbers 2 and 8 in the 31-40 and 41-50 age group.
- What would be limitations of this work, or possible countermeasures?
	- Can users detect suspicious background operations, e.g., from the decreasing battery life due to more power used while they are not actively using their smartphone?
- Due to the high complexity of the method, reproducing the work appears tedious and is unlikely to be successful given the lack of reported hyperparameters. Thus, providing the source code and, ideally, the collected datasets would significantly benefit this research direction and allow for proper validation.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an innovative cognitive-driven acoustic side-channel attack scheme called RecSpy, which records stereo tap acoustics on smartphones and exploits digit‑specific cognitive latencies via IRM, DDN, and an LLM‑based LIN to infer 6‑digit PINs on randomized keypads. Cross‑platform tests report strong Top‑K hit rates and "up to 4000x" gains over random guessing, indicating a practical threat.

### Strengths
1. Novel, cognition‑driven angle leveraging digit‑wise recognition latencies and age trends
2. Well‑structured pipeline: IRM (contrastive + frequency weighting), DDN (self‑supervised, conditioned alignment), LIN (LLM‑guided repetition priors and pruning).
3. Evaluations across multiple platforms (Android/iOS), devices, and environments (quiet/noisy) demonstrate the generalizability of RecSpy.

### Weaknesses
1. Lack of discussion on the influence of real-world variables.
2. Per‑subject PIN samples are limited (10 each), which may constrain robustness under fully randomized layouts.
3. No comparison to lightweight baselines or reporting of latency/energy for the LLM stack.

### Questions
1. The assumption of cognitive delay in the paper lacks sufficient verification. How will real-world variables, such as user input habits (e.g., typing rhythm, hesitation behavior), user's physical state (fatigue, tension), device hardware response delay, etc., affect cognitive delay, and thereby blurs the core basis for the model's inference?
2. In section 3.3, the paper mentions "143 participants" for latency observation, and in section 7.1, it mentions "40 volunteers" for evaluation. Are the training and evaluation participants entirely distinct?
3. While the paper compares RecSpy to random guessing, it does not benchmark against lightweight models (e.g., logistic regression, CRFs) or prior keystroke eavesdropping techniques. Is it possible that prior methods or lighter models can achieve attack performance slightly worse than RecSpy, but with much lower cost and constraints?
4. In section 6.3, the paper uses the premise of "frequently used duplicate digits" and then uses SSAST+Q‑Former+LLM to generate the "position pair duplicate prior matrix". What are the interpretable cues in the acoustic tokens that LLM uses to derive the "duplicate positions"? Is there a risk of overfitting to the training set's PIN distribution?
5. Although Section 7.2 provides Top‑K metrics and Section 7.5 further discusses "sustaining collection and continuous attempts", it does not  take into account typical retry limits or lockout strategies in real-world systems. What is the actual success rate in real-world systems (e.g., typically 3-10 attempts at most).
6. Figures 8/9/10/11 only show average results. What is the variance across participants?
7. There are some writing and formatting errors in the paper, for example:
    * The caption of Fig 3 only mentions "Aged 51 to 60", whereas Fig 3 displays four age groups.
    * A closing parenthesis ")" is missing in the right-hand label of Fig 3.
    * On line 363, "Pruning Reasoner" -> "Pruning Reasoner."

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes *RecSpy*, an acoustic side-channel attack for inferring PINs entered on randomized mobile soft keyboards by combining cognitively motivated digit-recognition latency patterns with spectrogram features and a two-stage reasoning pipeline. It introduces the IRM to learn user-specific recognition embeddings, the DNN to derive digit-level latency cues without explicit per-digit labels, and the LIN that uses an LLM to extract repetition priors and prune candidate PINs. The method targets a practical threat model in which a microphone-permitted background app records stereo keystroke acoustics on Android and iOS and performs offline inference, with evaluations across devices and indoor environments.

### Strengths
- The paper targets randomized soft keyboards and articulates a microphone-only threat model that avoids privileged sensors or external hardware.
- The architectural choices for the proposed modules are clearly described.
- The evaluation spans multiple devices and indoor settings and analyzes partial inference and repeated-digit priors.

### Weaknesses
- In this paper, the security significance is framed largely as a multiplicative gain over random guessing rather than attacker-centric success metrics (e.g., partial/average-case guessing entropy under retry limits). while Bonneau et al. recommend evaluating authentication schemes explicitly under throttled vs. unthrottled guessing to reflect real attack costs [1]. Reporting only something like "x improvement over $10^{-6}$" (e.g., the numbers reported in Section 7.2) can be somewhat misleading.

- Given the randomized layout, the paper should include a direct comparison against an audio-only baseline, such as a layout-agnostic KeyListener-like model [2] retrained to either classify each tap's spectrogram into digit labels without position cues or infer same/different constraints across taps, so the unique contribution of the cognition-driven modules is clear.

- Reliance on LLM reasoning to extract/prune repetition priors raises robustness and faithfulness concerns. Prior work shows chain-of-thought style reasoning can be brittle and unfaithful, and performance often depends on decoding strategies like self-consistency [3], which suggests the Pruning Reasoner may be sensitive to prompts/seeds and could overfit to training distributions unless carefully stress-tested.

- There are also reproducibility issues: the paper provides no stated plan to release code or data, and that the dataset is only described at a high level (devices, participant counts, and collection protocol) without an accessible corpus or retrieval details, preventing independent verification or re-analysis.

## References

- [1] J. Bonneau et al. The Quest to Replace Passwords. IEEE S&P 2012.
- [2] L. Lu et al. KeyListener: Inferring Keystrokes on QWERTY Keyboard of Touch Screen through Acoustic Signals. IEEE INFOCOM 2019.
- [3] X. Wang et al. Self-Consistency Improves Chain-of-Thought Reasoning in Language Models. ICLR 2023.

### Questions
- Under your stated threat model, what concrete evidence shows background audio capture is feasible during real PIN entry on modern Android/iOS?
- How sensitive is performance to device, microphone placement, environmental noise, and keypad randomization frequency? Where does the system fail?
- If code or data cannot be released, what exact artifacts (trained weights, prompts, logs, synthetic generators, evaluation scripts) will you provide to enable independent replication?

### Soundness
2

### Presentation
2

### Contribution
2
