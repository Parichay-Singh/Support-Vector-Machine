---


---

<p><strong>Support Vector Machines</strong></p>
<p><strong>What is it?</strong></p>
<p>Support vector machines (SVMs) are a particularly powerful and flexible class of<br>
supervised algorithms for both classification and regression. In this section, we will<br>
develop the intuition behind support vector machines and their use in classification<br>
Problems.</p>
<p><strong>What does it do?</strong></p>
<p>In case of Bayesian classification we learn a simple model describing the distribution of each underlying class, and use these generative models to probabilistically determine labels for new points. That was an example of generative classification; here we will consider instead discriminative classification: rather than modeling each class, we simply find a line or curve (in two dimensions) or manifold (in multiple dimensions) that divides the classes from each other.</p>
<p>As an example of this, consider the simple case of a classification task, in which the<br>
two classes of points are well separated</p>
<p><img src="https://lh3.googleusercontent.com/6qnxJYQ56NDLMSjD6YCUiivGs3Af9DI24A4WsoQlz9Z1KiGg-pPdEsUd7F4oEsIP0GP0Aj9n3EBb4NsRNtkZCtkvmN71wVbn70rWFagu4J0Li9vVRJxd9w0w64QIfhKqS2Uypw5a" alt=""></p>
<p>A linear discriminative classifier would attempt to draw a straight line separating the<br>
two sets of data, and thereby create a model for classification. For two-dimensional<br>
data like that shown here, this is a task we could do by hand. But immediately we see a problem: there is more than one possible dividing line that can perfectly discriminate between the two classes! We can draw them as follows:</p>
<p><img src="https://lh6.googleusercontent.com/gke2kaTq5mCRHm2tBx43PpL6fy--TZlYVGagOlO2F7MacytS2lJXOmulVnqCa632Rg5cptSjFuiJiTaJ6kvVM31l4ue1L8vBkfJBXcitHNl8YjPsPEqDJkuQx3ICM8hoXLfLxKFY" alt=""></p>
<p>These are three very different separators that, nevertheless, perfectly discriminate<br>
between these samples. Depending on which you choose, a new data point (e.g., the one marked by the “X”) will be assigned a different label! Evidently our simple intuition of “drawing a line between classes” is not enough, and we need to think a bit deeper.</p>
<p><strong>Maximizing The Margin</strong></p>
<p>Support vector machines offer one way to improve on this. The intuition is this:<br>
rather than simply drawing a zero-width line between the classes, we can draw<br>
around each line a margin of some width, up to the nearest point. Here is an example of how this might look:</p>
<p><img src="https://lh4.googleusercontent.com/WAb7RKSjz796v3TLuBp-Du6Q3ea4C1K9j8MRxlerThxIGMZe76eaF7oskQUUsfVRdty7E6sIVjdek1WeJxexIxAny-penhybiHSNuJQxPAupVgpC5eAIDz2aVD2obEpLXBGpFaNT" alt=""></p>
<p>In support vector machines, the line that maximizes this margin is the one we will<br>
choose as the optimal model. Support vector machines are an example of such a maximum margin estimator.</p>
<p>What we did was form an intuition of how SVM works. Lets see the outcome if we try to implement it using the sklearn library to train a model. For the time being, we will use a linear kernel and set the C parameter to a very large number(meaning of kernel &amp; parameter C shall be further discussed). This is how the outcome would look like:</p>
<p><img src="https://lh6.googleusercontent.com/SCYRzMcUm6quesJz-gTN_lLZQz4djAinPcOyKNqhDE5ngY8y0M8L51KOMd6DASS-2GIr9TsBeEIcPIbOsNI7Kg71f_SCbDmFH1qHCxZtn_kc7AaNj71LIidkwruGL_XpLihCGqY4" alt=""></p>
<p>This is the dividing line that maximizes the margin between the two sets of points.<br>
Notice that a few of the training points just touch the margin; they are indicated by<br>
the black circles in Figure. These points are the pivotal elements of this fit, and<br>
are known as the support vectors, and give the algorithm its name.</p>
<p>A key to this classifier’s success is that for the fit, only the position of the support vectors matters; any points further from the margin that are on the correct side do not modify the fit! Technically, this is because these points do not contribute to the loss function used to fit the model, so their position and number do not matter so long as they do not cross the margin.</p>
<p>In the left panel, we see the model and the support vectors for 60 training points. In<br>
the right panel, we have doubled the number of training points, but the model has<br>
not changed: the three support vectors from the left panel are still the support vectors from the right panel. This insensitivity to the exact behavior of distant points is one of the strengths of the SVM model.</p>
<p><img src="https://lh4.googleusercontent.com/iZemac0LUVtR0yEhrHGKF6gPXp9QdHKtwyZRYhD9t5wlbxk5R7e8KJjp8-N6yZ6xTiHt4VEodCF16bk2ohuGGm5Y3fEwFiITJj08kgexmM0iMM0fcuYiRNzoqW34nwTkUb32GTbz" alt=""></p>
<p>Beyond Linear Boundaries: Kernel SVM</p>
<p>What we made use of till now can be called a linear kernel as there was a linear boundary classifying the point into their various respective classes. But there can be data which is not linearly separable. For example:</p>
<p><img src="https://lh4.googleusercontent.com/YUzgsWFLkBHMl16jl-G4iQzl34r9hhRgock57hyhgXhGlkfRwT6kN0Dq1ujyNqQMsQ65zYG_N5nXCOMpmKsLQMHpGpgNe1OHJlUfnoABtHWmiDUZZN8dPp0aARfvxEweKv5UMiSs" alt=""></p>
<p>It is clear that no linear discrimination will ever be able to separate this data. But we<br>
can think about how we might project the data into a higher dimension such that a linear separator would be sufficient.</p>
<p>In this case, one simple projection we could use would be to compute a radial basis function centered on the middle clump. We can visualize this extra data dimension using a 3D plot.</p>
<p><img src="https://lh6.googleusercontent.com/7gRkMwGX5aByM5qDw8-CBle57fhUoWD-W--Zc_WziT6MdlHdQQunZHBnRFXdkfTwcW48VUkli1o1ytmPwLTXzWNNqHDRv-DdCimhlU_qgIYuvMC3tU138jjRp1xOdYBwktez8EYy" alt=""></p>
<p>We can see that with this additional dimension, the data becomes trivially linearly<br>
separable, by drawing a separating plane at, say, r=0.7.</p>
<p>Here we had to choose and carefully tune our projection; if we had not centered our<br>
radial basis function in the right location, we would not have seen such clean, linearly<br>
separable results. In general, the need to make such a choice is a problem: we would like to somehow automatically find the best basis functions to use.</p>
<p>One strategy to this end is to compute a basis function centered at every point in the<br>
dataset, and let the SVM algorithm sift through the results. This type of basis function<br>
transformation is known as a kernel transformation, as it is based on a similarity relationship (or kernel) between each pair of points.</p>
<p>A potential problem with this strategy—projecting N points into N dimensions—is that it might become very computationally intensive as N grows large. However,because of a neat little procedure known as the kernel trick, a fit on kernel transformed data can be done implicitly—that is, without ever building the full N dimensional representation of the kernel projection! This kernel trick is built into the SVM, and is one of the reasons the method is so powerful.</p>
<p>In sklearn, we can apply kernelized SVM simply by changing our linear kernel to an RBF (radial bias function) kernel, using the kernel model hyperparameter. Using this kernelized support vector machine, we learn a suitable nonlinear decision boundary. This kernel transformation strategy is used often in machine learning to turn fast linear methods into fast nonlinear methods, especially for models in which the kernel trick can be used.</p>
<p><img src="https://lh4.googleusercontent.com/iGiaWyvCl4LRZpnMh_zQmCjPcOa1DwyfOTYaG8_jmn1jiacJloA7bLecDDxk8yQuLhFcpRbl7WoUJMxScfq1Jrb0V4C18JFUX5bIYVyCtbBXTTgLUStUya8QZgrgKhoLq1KNWiN9" alt=""></p>
<p><strong>Tuning SVM: Softening Margins</strong></p>
<p>Our discussion so far has centered on very clean datasets, in which a perfect decision boundary exists. But what if your data has some amount of overlap? For example, you may have data like this:</p>
<p><img src="https://lh4.googleusercontent.com/k3s_RYOkxvGgg7PecI6IrEoHXK4MU1YBpCa-KybPvASK2-U2hMqkn_gJ8SNKYXoiMzphydG54Gc1yA3lxmvsRQQLPJlhiONiJN9n8uHfU7yFe4hl-32H49c8UeVmLn3rADbzceLo" alt=""></p>
<p>To handle this case, the SVM implementation has a bit of a fudge-factor that “softens” the margin; that is, it allows some of the points to creep into the margin if that allows a better fit. The hardness of the margin is controlled by a tuning parameter, most often known as C. For very large C, the margin is hard, and points cannot lie in it. For smaller C, the margin is softer, and can grow to encompass some points. The plot shown gives a visual picture of how a changing C parameter affects the final fit, via the softening of the margin.</p>
<p><img src="https://lh6.googleusercontent.com/yYS1NIh82MODKvEQHl_7AbReVo1p0bMj6qvdEL1v01IITSZGa5Ymo55fK__EmPT0krVSlndpobiJtKBxdlbIaDomO3WcAvkJrxs8ZjkU8UKWfrPfUP1BpEDcx5HwhKuhABnDXpru" alt=""></p>
<p>The optimal value of the C parameter will depend on your dataset, and should be<br>
tuned via cross-validation or a similar procedure.</p>

