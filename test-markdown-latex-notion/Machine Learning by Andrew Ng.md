# Machine Learning by Andrew Ng

Week 1 

Introduction

**Definition:** Machine Learning is the practice of teaching a computer to learn via trials to execute tasks without being programmed. Repeat Experience E, check Performance Measure P, execute task T

**Applications:**

- Data Mining: user trends
- Unprogrammable applications: autopilot, navigation, voice recognition, computer vision
- Recommendation

**Types:**

- Supervised learning: predict new results based on a given data set
    - predict *continuous* valued output based on a given set of *labelled* data, aka. *Regression Problem*
    - predict *discrete* valued output based on a given set of *labelled* data, aka. *Classification Problem*
    - predict output based on a given set of *labelled* data having *infinite* features, aka. *Support Vector Machine*, i.e. predict the weather based on wind, sun, cloud, earth rotation...
- Unsupervised learning
    - Cluster input data without any accompanying labels based on their relationships , no feedbacks are applied on prediction results
    - Clustering algorithms
    - i.e. google news, human genes classification, server clustering, cohesive groups of social network friends, market segmentations, astronomical data analysis, cocktail party voice separation
- Recommender Systems
- Reinforcement Learning

One Variable Linear Regression

**Model and Cost Function**

Model representation: use *training set* to find out *hypothesis function*

![Machine%20Learning%20by%20Andrew%20Ng/Untitled.png](Machine%20Learning%20by%20Andrew%20Ng/Untitled.png)

Example: linear regression model hypothesis

$$h_\theta(x)=\theta_0+\theta_1x$$

Goal: minimize cost function

$$J(\theta_0, \theta_1)=\min_{\theta_0,\theta_1}\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2=\min_{\theta_0,\theta_1}\frac{1}{2m}\sum_{i=1}^m(\theta_0+\theta_1x^{(i)}-y^{(i)})^2$$

Methods to find the hypothesis:

- Normal function
- Gradient descent (1)
- Conjugate gradient (2)
- BFGS (2)
- L-BFGS (2)

/tab

Advantages of (2):

- alpha is tuned and chosen automatically by the algorithms
- Often swifter than gradient descent

Disadvantages of (2):

- More sophisticated

**Gradient Descent**: a method to find minima of cost function

$$\text{repeat until convergence}\ \left\{\theta_i=\theta_i-\alpha\frac{\partial}{\partial\theta_i}J(\theta_0,\theta_1)\right\},i=\overline{0,1},\ \alpha>0:\text{learning rate}$$

![Machine%20Learning%20by%20Andrew%20Ng/Untitled%201.png](Machine%20Learning%20by%20Andrew%20Ng/Untitled%201.png)

Cost function of *linear regression* always provides a global minima. Learning rate *alpha* must not be too large to avoid divergence, and can be fixed throughout the steps.

**Parameter learning**

Week 2

Multi Variable Linear Regression

Hypothesis model function:

$$h_\theta(x)=\theta^Tx =\left[\begin{matrix}\theta_0 & \theta_1 & \dots & \theta_n\end{matrix}\right]\left[\begin{matrix}x_0 \\ x_1 \\ \vdots \\ x_n\end{matrix}\right],\text{with}\  x_0=1$$

Two methods:

- Normal function: no need to scale feature values but computation cost O(n3) is large, i.e. n = 10,000

$$\theta=(X^TX)^{-1}X^Ty$$

- Gradient descent: the goal is to minimize cost function. Works well if n is large, with computation cost O(kn2)

$$J(\theta_0, \theta_1,\ldots,\theta_n)=\min_{\theta_i}\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$$

$$\text{repeat until convergence}\ \left\{\theta_j=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})\cdot x_j^{(i)}\right\}\ ,j=\overline{0,1}$$

**Computing Parameters Analytically**

*Feature scaling:* if x are on a similar scale of largeness, the cost function have its graph in circular layers, which allows the gradient descent to converge faster. Conversely, if x are two different, the cost function has elliptic layers, taking longer time to converge. The goal is to get every features into an approximately small range, i.e. [-1, 1]

![Machine%20Learning%20by%20Andrew%20Ng/Untitled%202.png](Machine%20Learning%20by%20Andrew%20Ng/Untitled%202.png)

*Mean normalization:* 

$$x_i=\frac{x_i-\overline{x}}{\max{x_i}-\min{x_i}}\in[-0.5,0.5]$$

*Choose learning rate*

- large alpha makes the gradient descent diverge
- sufficiently small alpha helps it converge
- too small alpha makes convergence slow
- trials: 0.001, (0.003), 0.01, (0.03), 0.1, (0.3), 1, etc and plot J(θ)

*Features and Polynomial Regression*

- Combine features into one, i.e. length x width = area
- Consider high exponential components as new features and **normalize** them

$$h_\theta(x)=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3 \\ h_\theta(x)=\theta_0+\theta_1x+\theta_2x^2=\theta_0+\theta_1x_1+\theta_2x_2 \\ h_\theta(x)=\theta_0+\theta_1x+\theta_2\sqrt{x}=\theta_0+\theta_1x_1+\theta_2x_2$$

Week 3

[Logistic Regression](https://d3c33hcgiwev3.cloudfront.net/_964b8d77dc0ee6fd42ac7d8a70c4ffa1_Lecture6.pdf?Expires=1584403200&Signature=f56LGOSDpzf4rk2h1dy6t3tbURfDVhpA9RW-6kacwYbLT514CINrSa8KfJCAuSQkzVlFzIMqjIfGQEcy0adLgBnbyiOjXAYLno4DA3nbcK4SoqVtXKAJPL~bGR-4bPbK1c6gJcux58jvJ1B5SmncN-T56on1BNqqU~d2xydQFHY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

- Classification problem, h in [0, 1]
- Focus on **binary** classification problems

Hypothesis representation: sigmoid/logistic function = probability that y = 1 on input x (either linear or nonlinear)

$$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}=P(y=1|x;\theta)$$

Decision boundary

- h(z) ≥ 0.5 → z ≥ 0
- h(z) < 0.5 → z < 0
- Do *not* depend on **training set** but the hypothesis function and its parameters thetas
- The training set can be used to fit the theta. Once we have the thetas, that are what define the decision boundary

![Machine%20Learning%20by%20Andrew%20Ng/Untitled%203.png](Machine%20Learning%20by%20Andrew%20Ng/Untitled%203.png)

![Machine%20Learning%20by%20Andrew%20Ng/Untitled%204.png](Machine%20Learning%20by%20Andrew%20Ng/Untitled%204.png)

Cost function:  as sum-of-square cost function returns one global minima and multiple local minimas (non-convex), logarithmic cost function provides only a global minima (convex):

$$J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\ \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; \text{if y = 1} \\ \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; \text{if y = 0}$$

$$\mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \newline \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0$$

$$\Rightarrow \mathrm{Cost}(h_\theta(x),y) = -y\log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

$$\Rightarrow J(\theta)=-\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}\log(h_\theta^{(i)}(x)) + (1 - y^{(i)}) \log(1 - h_\theta^{(i)}(x))\right]$$

$$\mathrm{Repeat} \left\{ \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \right\}$$

Vectorization:

$$h = g(X\theta) \\ J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \newline \theta=\theta-\alpha\cdot\frac{1}{m}\cdot X^T\left\{g(X\theta)-\vec{y}\right\}$$

Advanced Optiomization:

- Conjugate gradient
- BFGS
- L-BFGS

Write a function returning both cost function and its partial derivatives:

    function [jVal, gradient] = costFunction(theta)
      jVal = [...code to compute J(theta)...];
      gradient = [...code to compute derivative of J(theta)...];
    end
    
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    initialTheta = zeros(2,1);
       [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

**Multiclass classification**

- Classify emails into Inbox, Social Networks, Promotions
- Use one vs all method
    - Output classes: y = {0, 1, 2, ..., n}
    - Train n + 1 logistic regression classifiers h_theta(x)^{(i)}

    $$h_\theta^{(i)}(x) = P(y = i | x ; \theta) \\ \mathrm{prediction}=\max_{i=\overline{0,n}}h_\theta^{(i)}(x)$$

Solving the Problem of Overfitting: Regularization

Overfitting: perfect TBC

Week 4 & 5

Neural Networks

Representation

Learning

Week 6

Advice for Machine Learning

Machine Learning System Design

Week 7

Support Vector Machine

Week 8

Unsupervised Learning

Dimensionality Reduction

Week 9

Anomaly Detection

Recommender Systems

Week 10

Large Scale Machine Learning

Week 11

Example: OCR