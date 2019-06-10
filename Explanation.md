# Computing the Triangle Correlation Function

The triangle correlation function is a special case of the 3-point correlation function, which is defined as the inverse Fourier transform of the bispectrum: 
$$
\begin{equation}
    \Xi(\mathbf{r,s}) = \frac{V^2}{(2\pi)^{2D}}\int\int d^Dk \ \ d^Dq \ e^{i(\mathbf{k \cdot r + 	q\cdot s})} \mathcal{B}(\mathbf{k,q}) \ \ .
\end{equation}
$$
In the triangle correlation function, we let ***r*** and ***s*** form the legs of an equilateral triangle. In order to do this, we simply define ***s*** to be ***r*** rotated by 60 degrees. In this way, the 3-point correlation function becomes a function of ***r*** only. Additionally,  if we define:
$$
\mathbf{p} = \begin{bmatrix}
			k_x + \frac{1}{2}q_x + \frac{\sqrt3}{2}q_y \\
			k_y - \frac{\sqrt3}{2}q_x + \frac{1}{2}q_y \\
            k_z + q_z
			\end{bmatrix} \ \ ,
$$
then 
$$
\mathbf{k\cdot r + q\cdot s = p \cdot r} \ \ .
$$
We may take a rotational average, introduce a window function, and discretise the integral for numerical computation to arrive at: 
$$
\xi_3(r) = \sum_k \sum_q \omega_D (pr) \frac{\mathcal{B}(\mathbf{k,q})}{|\mathcal{B}(\mathbf{k,q})|} \ \ ,
$$
where we consider only the phase factor of the bispectrum rather than its full form, and the window function is defined as: 
$$
\omega_D(x) = \begin{cases}
\frac{\sin x}{x} \quad \ \ \text{for} \ \ D = 3\\
J_0(x) \quad \text{for}  \ \ D=2
\end{cases} \ \ .
$$
Introducing upper limits on the norms of ***k*** and ***q*** in order to avoid divergence of the sum, and a pre-factor since for extreme cases discretisation no longer holds [^1] , we get the final expression for the triangle correlation function which will be used in the computation : 
$$
s(r) = \bigg(\frac{r}{L}\bigg)^{3D/2} \sum_{k,q \leq \pi/r}\omega_D(pr) \frac{\mathcal{B}(\mathbf{k,q})}{|\mathcal{B}(\mathbf{k,q})|} \ \ .
$$
For the rest of this document, I will explain the code I have written for computing the triangle correlation function for a two dimensional field, since some of the steps are not very intuitive. 



![](vects.gif)

[^1]: For more detail, see Gorce and Pritchard papaer.



