# Conic Linear Units: Orthogonal Equivariance Improves General-Purpose Nonlinearities

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Most activation functions operate component-wise, which restricts the equivariance of neural networks to permutations. We introduce Conic Linear Units (CoLU) and generalize the symmetry of neural networks to continuous orthogonal groups. By interpreting ReLU as a projection onto its invariant set—the positive orthant—we propose a conic activation function that uses a Lorentz cone instead. Its performance can be further improved by considering multi-head structures, soft scaling, and axis sharing. CoLU associated with low-dimensional cones outperforms the component-wise ReLU in a wide range of models—including MLP, ResNet, and UNet, etc., achieving better loss values and faster convergence. It significantly improves diffusion models' training and performance. CoLU originates from a first-principles approach to various forms of neural networks and fundamentally changes their algebraic structure.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new activation function called CoLU. In contrast to ReLU, CoLU is not component-wise but is inspired by physics to preserve symmetries. To do so it mixes information between components while retaining an O(n) runtime for no efficiency overhead. The paper shows that CoLU leads to either faster or equivalent training speed compared to ReLU on a variety of tasks while improving generalization and reducing overfitting.

### Strengths
The paper proposes (as far as I know) an original approach to non-point-wise activation functions in neural networks. The mathematical basis of preserving symmetries, and the references to geometry to get there, are grounded and worthwhile. Specifically the authors point out a common assumption machine learning practitioners make---that permutation symmetry is important in an activation function---and explore an alternative choice. This is a good model for basic research. Because CoLU has the same runtime as ReLU, its significance could be high if it demonstrably improves training speed (Fig 6) or other characteristics like generalization (Fig 10) on tasks at scale.

### Weaknesses
The main weakness for me is skepticism that the experiments are showing CoLU's value over ReLU or other point-wise activation functions. I'll dive into that, and separately I will provide a few examples of where the exposition confused me in the hopes the authors can benefit from this information.

1. Experiments: I am fascinated by CoLU so would appreciate seeing more rigorous experiments and reporting demonstrating its value.

    a. Thank you for reporting exact details of your dimensions and hyperparameters. That's good. For one thing, I would like to see experiments at a bigger scale. The experiments I see use hidden dimension 20 (for VAE) and 256 (for GPT). I believe your experiments would be more rigorous if they tried larger scales than these. There's no specific scale that becomes "rigorous" but trying a few scales across a few orders of magnitude (say parameters = 1M, 10M, 100M) would go a long way to showing that results are trends, not flukes. Keeping the same modalities would be fine, and it is helpful that you train across multiple modalities so you can demonstrate alignment or other gains in a variety of settings. I see some large-scale experiments on diffusion models (835M parameters, from line 1004), but I don't see any training curves, figures, or results at all for these except for the generated pictures.

    b. Please be careful when reporting results. In Table 4, thank you for reporting standard deviations. That is very helpful. That said, the table shows CoLU outperforming ReLU with a margin that is small enough to be well within one standard deviation (diff = 0.0036, stdev = 0.0100). This difference means the results are not conclusive. Could you either run more seeds, run a bigger scale where the difference resolves, or change your table's caption from "CoLU outperforms ReLU" to "CoLU performs on par with ReLU" or a related phrasing? For another example, in Table 6, you bold the better result on eval loss for CoLU but do not bold the other column which shows a better result on train loss for ReLU. This makes the table look skewed that CoLU is better, when in fact it is a toss-up. Finally, I'm confused by Table 3. The last two rows look identical except for the width C changing from 512 to 511. Why do you make this particular comparison?

In the end, it will be important for you to establish that CoLU in fact outperforms ReLU, or that it has some other value aside from a nice theoretical basis. Simply extending the generality of a function class is interesting, but not worthwhile in and of itself if there aren't reasons to make the generalization.

2. In Fig 1, it would help to define C, G, and what you mean by maximum norm threshold in the caption or elsewhere. At present you do not define C in general anywhere, for example. You use it in specific cases like "In this illustrative example, the network width is C=6." I think Fig 1 has potential to communicate lots of intuition about CoLU but it is confusing in a few ways at the moment. Also, you may be able to omit the blue circles since they are repeated and could distract from the main point, which is the activation function.

3. Typos. Please fix typos in your paper before submitting. Examples: "orthogornal" in Lemma 4.8, and the cut-off sentence on line 280 that reads "This is unprecedented in component-wise activation networks whose s." Also the grammar in the statement of Lemma 4.9 (the comma, as written, should be a period). This is not as important as earlier points, but it's helpful for the reader if the paper is well proofread.

### Questions
I've listed several questions in the "weaknesses" section already. For a few more:

1. How does CoLU perform on a larger language modeling experiment? For a simple test, you can try NanoGPT for instance (https://github.com/karpathy/nanoGPT), replace ReLU with CoLU, and compare. Once again, I'd find it more convincing if it's a larger dataset than Shakespeare which has only a few million characters total. For instance you can use FineWeb10B.

2. Your nicely titled section "Why Conic Activation Functions" does not answer this question for me. Something like Lemma 4.8 gets close to why conic projections are nice, but it feels not fully explained for such an important motivation for your entire choice of CoLU. The line "we assume there are low-dimensional block sub-spaces that can hold orthogonal equivariance" also seems important for this lemma, yet I am left wondering why we should assume this, or how assuming it helps us. I would appreciate a more clarifying explanation of why conic projections are good for deep learning. Otherwise, I feel like what I am currently getting is a section that could be titled "What are Conic Activation Functions."

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper introduces Conic Linear Units (CoLU), which bring orthogonal group symmetry to neural networks. CoLU surpasses state-of-the-art component-wise activation functions like ReLU.

### Strengths
- The paper introduces a non-component-wise activation function (CoLU) that broadens neural network symmetry to continuous orthogonal groups, marking a new direction in the field. A novel method.

- The proposed activation CoLU demonstrates strong performance across multiple architectures, with notable improvements over ReLU in various deep learning models.

- The derivation for theoretical aspects of CoLU is well-written and accurate.

### Weaknesses
- The paper claims that CoLU's computational complexity is negligible compared to matrix multiplications, but I question if this holds true for large-scale models since it is *non-component-wise*. While I understand that the rebuttal phrase is only 7-day, which is short for a large experiment, I believe this is a crucial aspect for a methodology paper to gain acceptance.

- Minor typos:

line 19: models’s -> models

line 31: Convolution layers -> Convolutional layers 

line 40: equivarance -> equivariance

line 53:
Can forms of equivariance more general than permutation improve neural networks? -> I really don't understand what this question means.

### Questions
- Please conduct a large-scale experiment to compare the performance and computational time between CoLU and ReLU. Ensure that you report the results, including performance and computational time, and the experimental setup in detail. I consider this question significant, and as it addresses the paper’s gap, I am willing to raising the score for acceptance if answered comprehensively.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a new activation function called conic linear units (CoLU), which applies block-wise 
 non-linearity to neurons. Specifically, CoLU projects a set of neurons to a light cone.
The authors experimented with this activation in various tasks such as synthetic image classification, image generation with diffusion models, and different architecture types such as MLP, ResNet, and UNet.

### Strengths
- The construction of the proposed activation is unique. It is motivated by spacetime symmetry, and basically, any activation functions can be replaced with the proposed one.
- The authors tested CoLU on various tasks, from synthetic data reconstruction to image diffusions.
- The authors investigated the linear mode connectivity property and observed the CoLU-based models also exhibit the property similarly to ReLU-based networks, which is intriguing.

### Weaknesses
- It is hard to judge the significance of CoLU's performance as the paper limits its comparisons to its own experimental models without referencing results from other studies. The following are more specific concerns about performance comparison.
  - Regarding ResNet-56 experiments, the reported test error rate of ResNet56 in Table 4 is ~9％, but the same model achieves ~7% error rate in the original ResNet paper [1]. This makes me concerned about the validity of this experiment. Clarify if there are any differences in implementation or training settings compared to the original ResNet paper that could explain the performance gap.
  - The performance of toyMLP is quite low. I quickly experimented with ReLU-based 512-dim two-layer MLP on MNIST (the same architecture used in the paper), and it got 98% test accuracy, which is much higher than the scores shown in Table 3. The authors should explain this performance gap. The code I used is shown below. 
  - In the abstract and the introduction, the paper says that the CoLU activation significantly outperforms ReLU-based diffusion, but this statement is not fully validated in the experimental section. Table 5 only presents the training loss and does not show conventional image generation metrics such as FID or test ELBO, which should be included in the table to validate the statement.

- It’s unclear how the equivariance property of the proposed activation relates to overall network performance, expressivity, and generalization.
The performance improvement in Experiment 5.1 makes sense since the data has rotation symmetry, while the other tasks and datasets do not involve rotation symmetry. It would be good for the authors to explain why the proposed activations contribute to performance improvement on such datasets and tasks without rotational symmetry. The authors might be trying to explain the relationship through the theoretical part (section 4), but the current version is so complex that it's hard to understand what is the main contribution of this section. I would appreciate it if the authors could provide a more detailed summary of this section.

[1] https://arxiv.org/abs/1512.03385
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Training settings
batch_size = 128
test_batch_size = 1000
epochs = 5
lr = 0.001
no_cuda = False
seed = 1
log_interval = 10

use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

### Questions
- Regarding the linear mode connectivity.
Are CoLU models better in terms of the mode connectivity? For example, I believe the authors only show the loss barrier plot of CoLU's networks in Figure 14. Can you include loss barrier plots for both CoLU and ReLU models in the figure for direct comparison?

Typos
- l280: Seems texts missing after `whose s`.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new activation function, the conic linear unit (CoLU). Inspired by the conic projections in relativistic geometry, the CoLU builds in additional equivariance properties compared to the ReLU activation function by acting on groups of neurons instead of the typical per-neuron activation. In particular, CoLU supports orthogonal equivariance within a feature vector (or subvector/group of features), i.e., equivariance to a rotation around a conic axis. 

The authors develop mathematical proofs of the additional equivariance properties and provide empirical results to show an improved generalization ability compared to ReLU by way of improved performance in toy settings, ResNets for classification on CIFAR-10, and UNETs for image generation.

### Strengths
- I found the proposed approach for baking equivariance properties directly into the activation function original. The projection onto a cone to impose rotational equivariance makes sense and may be useful in many settings.
- The paper provided experimental evidence for improved performance in many settings, which could point to a potentially general contribution to the field of deep learning.
- The paper was written well with clear and concise statements and was a pleasure to read.

### Weaknesses
I have several concerns that contributed to my moderate score for the paper. I elaborate below.
- I didn't see any strong evidence in the paper that real-world feature spaces contain rotationally symmetry, so I find it hard to appreciate why imposing this additional capability should be expected to be beneficial a priori. The theoretical proofs show that orthogonal equivariance is an additional feature of the architecture with CoLU, but not that this is necessary in practice or under some assumptions about the feature space.
- The performance improvements are modest, about 1% on CIFAR and MNIST. Furthermore, table 4 shows a lower training loss for CoLU in addition to the higher test accuracy. One possible explanation for this would just be that the ReLU takes longer to converge with the chosen hyperparameters. I would suggest training the ReLU network longer up to the same training loss if possible to make the claim about better generalization properties stronger.
- Minor: There is no detailed analysis or empirical proof that the training and inference speeds remain unaffected. Adding this would strengthen the claims of the paper since the proposed activation function is significantly more complex than the elementwise ReLU.

### Questions
- Is the link to relativistic geometry meaningful or relevant in any way? The light cone and plane of simultaneity concepts are introduced but don't seem to be relevant to the development of the CoLU activation function or the rest of the paper. It seemed to me that the actual maths could be presented more directly with standard linear algebra and group theory, but perhaps I am wrong. In any case, I would suggest removing them or shortening this material for the sake of clarity since they add unnecessary complexity and distraction.
- Soft projection and axis sharing were introduced but were only tested in toy settings. If those results hold, then wouldn't the best performance be obtained by including them in the large-scale experiments? Why was this not included?

### Soundness
3

### Presentation
2

### Contribution
4
