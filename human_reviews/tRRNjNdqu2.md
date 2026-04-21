# Self, Semi and Fully Supervised Training for Autoencoders using Ternary Classification

- Avg Score: 2.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 3

## Abstract
Autoencoders are usually trained in a self-supervised fashion. In the context of anomaly detection, research shows that they can also be trained in a fully supervised one, using binary class labels, namely HEALTHY and FAULTY. However, when working with real world data, such an approach might not be suitable. It is hard to binary classify data coming from equipment that has been in operation for a long time, is affected by wear and tear. In additional, its real current health status is unknown. Moreover, historical data is not usually labeled, and only maintenance interventions are recorded. To alleviate this problem, a third label is introduced, UNKNOWN, which enables the autoencoder to learn the structure of healthy and faulty data from the correspondingly labelled data points. This structure is used in reconstructing the UNKNOWN inputs. This can increase the performance of autoencoders in a wide range of anomaly detection cases, especially when the timeseries data used to train the autoencoder comes from machines that have been in operation for a long time. This is especially relevant in the case of industrial machinery. Different label-aware loss functions which can enable the training of an autoencoder, using the three aforementioned labels, in any combination of self, semi and fully supervised training are investigated in this work. The loss functions presented in this paper enable an autoencoder to achieve particularly good anomaly detection performance on a clutch-slip detection dataset acquired from a test bench which simulates the drivetrain of an electric Range Rover Evoque. The dataset is presented in the appendix.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies a third label (UNKNOWN) that enables the autoencoders to learn the structure of healthy and faculty data from the correspondingly labeled data points.

### Strengths
NA.

### Weaknesses
1. The technical novelty is very limited. And the proposed method does not make sense to me. It is hard to read the paper. 
2. I did not get the whole point of the motivation and practical meaning of the paper. Perhaps it needs a better presentation. 
3. They only evaluated on one dataset, which is not convincing enough. Also it is hard to interpret the table. 
4. Figure 4 should be of higher resolution.

### Questions
See weaknesses.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose labeled-aware loss functions for autoencoders.
Based on 4 baseline loss functions, MSE, RMSE, L2-norm and
Squared-L2-norm, they introduce additional parameters depending on the
labels (healthy, faulty, and unknown).  One approach is to include the
parameter as an exponent (exponent-based), another is to multiple by
the parameter (weight-based).  To prevent negative loss values, they
use a soft limit by reducing the absolute values.

Using a dataset that contains parts that are healthy, faulty, and
unknown condition, they set up different scenarios for supervised,
semi-supervised, and unsupervised learning.  Label-aware loss
functions seem to perform better.

### Strengths
The proposed idea is relatively straight forward.  A real-world dataset
from automobile parts is used.

### Weaknesses
The 4 proposed baseline loss functions seem to be very similar in
terms of minimization.  Particularly, MSE and Squared-L2 norm differ
by a constant factor, which can be absorbed by the learning rate.
Autoencoders can be used for learning features without labels.
However, the authors propose modifying the autoencoder loss functions
with labels, which makes learning the autoencoder supervised.  Then a
comparison with regular supervised learning without an autoencoder
would be important.

Details are in questions below.

### Questions
1.  Table 1: Squared-L2 norm: the square and square root can be
cancelled out.  Squared-L2 norm / n is equivalent to MSE, so the two
loss functions differ by a constant (n is the same in both loss
functions).  That is, when Squared-L2 norm is minimized, MSE is
minimized.  When MSE is minimized, RMSE is also minimized.  Similarly,
when Squared-L2 norm is mininized, L2-norm is minimized.  Hence, the
four loss functions seem to be equivalent in terms of minimization.
Any insights on why these four loss functions are chosen?


2.  Table 2: $e_la$ and $e_a$ are different, but only $e_la$ is
discussed in the text.  What are the values for $e_a$?


3.  Equation 1: what is the motivation?


4.  Table 4: What are H, HU, HUF?  Does each file have multiple
instances?

5.  Tables 6 and 7: how do you get inf % improvement?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies using autoencoders to perform anomaly detection, and the central observation is that labeled data with unknown type can improve model performance.

### Strengths
The paper made certain observations that have practical values.

### Weaknesses
The technical novelty is insufficient for ICLR.  The techniques proposed in the paper seem to be standard.

### Questions
NA

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
