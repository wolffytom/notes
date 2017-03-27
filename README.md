学号:1601210300 姓名:崔家梁 深度学习课3.28日作业-BP算法求梯度公式推导

###1 符号记法
使用上课所用的ppt和书中的符号记法  
该网络一共 $l$ 层, 激活函数为$f$  
第 $k$ 层 ($k=1,2,...l$) 的输入为 $h^{(k-1)}$ ,($h^{(k-1)} \in R^{(m_{k-1} \times 1)}$)  
中间结果为 $a^{(k)}$  ,输出为 $h^{(k)}$,  
$a^{(k)} = b^{(k)} + W^{(k)}h^{(k-1)} \tag{1.1}$ $h^{(k)} = f(a^{(k)}) \tag{1.2}$
(其中$a^{(k)},b^{(k)},h^{(k)} \in R^{(m_{k} \times 1)};W^{(k)} \in  R^{(m_k \times m_{k-1})}$)  
该网络的最终输出为误差项$J = L( \hat{y} , y ) + \lambda \Omega (\theta) \tag{1.3}$
(其中$J \in R, \hat{y} = h^{(l)})$  

###2 对于参数$W^{(k)}$
对于梯度 $\nabla_{W^{(k)}}J$ ,根据链式求导法则有:
$$\nabla_{W^{(k)}}J
= \sum_j (\nabla_{W^{(k)}}a^{(k)}_j) \dfrac{\partial J}{\partial a^{(k)}_j}$$
其中 $a^{(k)}_j$ 为 $a^{(k)}$ 的第 $j$ 个分量, $W^{(k)}_i$ 为$W^{(k)}$ 的第 $i$ 行, 对于 $a^{(k)}_j$ 有:
$$a^{(k)}_j = W^{(k)}_j h^{(k-1)}$$
由于 $a^{(k)}_j$ 只与 $W^{(k)}$ 的第 $j$ 行有关,所以有:
$$\nabla_{W^{(k)}_i} a ^{(k)}_j =\begin{cases}
0,i \neq j\\
(h^{(k-1)})^T,i=j
\end{cases}$$
因此对于矩阵形式的 $W^{(k)}$ 有:
$$\nabla_{W^{(k)}}a^{(k)}_j
=((\nabla_{W^{(k)}_1}a^{(k)}_j)^T,(\nabla_{W^{(k)}_2}a^{(k)}_j)^T,...,\nabla_{W^{(k)}_j}a^{(k)}_j)^T,...,(\nabla_{W^{(k)}_{m_k}}a^{(k)}_j)^T)^T
=(0,...,0,(h^{(k-1)})^T,0,...,0)^T\\
=e^{(j)}(h^{(k-1)})^T
,(其中e^{(j)}是第j个元素为1,其余元素为0的向量)$$
因此,
$$\nabla_{W^{(k)}}J
= \sum_j (\nabla_{W^{(k)}}a^{(k)}_j) \dfrac{\partial J}{\partial a^{(k)}_j}
= \sum_j (e^{(j)}(h^{(k-1)})^T) \dfrac{\partial J}{\partial a^{(k)}_j},(\dfrac{\partial J}{\partial a^{(k)}_j} \in R)\\
= (\sum_j \dfrac{\partial J}{\partial a^{(k)}_j} e^{(j)}) (h^{(k-1)})^T
= (\nabla_{a^{(k)}}J)(h^{(k-1)})^T \tag{2.1}$$
###3 对于参数$b^{(k)}$
$$\nabla_{b^{(k)}}J
=(\dfrac{\partial a^{(j)}}{\partial b^{(j)}})^T \nabla_{a^{(k)}}J$$
上式中 $\dfrac{\partial a^{(j)}}{\partial b^{(j)}}$ 是Jacobian矩阵,则有:
$$\dfrac{\partial a^{(j)}}{\partial b^{(j)}}=I$$
因此有:
$$\nabla_{b^{(k)}}J
=\nabla_{a^{(k)}}J \tag{3.1}$$
###4 递推关系
结合(2.1)(3.1),令 $\delta^{(k)} = \nabla_{a^{(k)}}J$ ,有:
$$\begin{cases}
\nabla_{W^{(k)}}J=\delta^{(k)}(h^{(k-1)})^T\\
\nabla_{b^{(k)}}J=\delta^{(k)}
\end{cases} \tag{4.1}$$
$$\nabla_{a^{(k)}}J
=(\dfrac{\partial a^{(k+1)}}{\partial a^{(k)}})^T\nabla_{a^{(k+1)}}J \tag {4.2}$$
而:
$$\dfrac{\partial a^{(k+1)}}{\partial a^{(k)}}
=(\dfrac{\partial a^{(k+1)}}{\partial h^{(k)}})(\dfrac{\partial h^{(k)}}{\partial a^{(k)}}) \tag{4.3}$$
对于Jacobian矩阵 $\dfrac{\partial a^{(k+1)}}{\partial h^{(k)}}$ ,根据定义,有
$$a^{(k+1)}_i = W^{(k+1)}_{i,j} h^{(k)}_j$$
$$\dfrac{\partial a^{(k+1)}_i}{\partial h^{(k)}_j}=W^{(k+1)}_{i,j}$$
$$\dfrac{\partial a^{(k+1)}}{\partial h^{(k)}}=W^{(k+1)} \tag{4.4}$$
对于Jacobian矩阵 $\dfrac{\partial h^{(k)}}{\partial a^{(k)}}$,有:
$$\dfrac{\partial h^{(k)}_i}{\partial a^{(k)}_j}
= \begin{cases}
0,i \neq j\\
f'(a^{(k)}_j), i=j
\end{cases}$$
因此有
$$\dfrac{\partial h^{(k)}}{\partial a^{(k)}}
=diag(f'(a^{(k)}_i))=f'(a^{(k)})I,(其中f'(a^{(k)})=(f'(a^{(k)}_1),...f'(a^{(k)}_{m_k}))^T) \tag {4.5}$$
结合(4.2)(4.3)(4.4)(4.5),有
$$\delta^{(k)}=\nabla_{a^{(k)}}J
=(W^{(k+1)}(f'(a^{(k)})I))^T \delta^{(k+1)}
=f'(a^{(k)})I(W^{(k+1)})^T \delta^{(k+1)}\\
=f'(a^{(k)}) \odot ((W^{(k+1)})^T \delta^{(k+1)}) \tag {4.6}$$
###5 初始值与算法
$$\delta^{(l)}=\nabla_{a^{(l)}}J=f'(a^{(l)}) \odot \nabla_{h^{(l)}}J
=f'(a^{(l)}) \odot \nabla_{\hat{y}}J
\\=f'(a^{(l)}) \odot \nabla_{\hat{y}} L(\hat{y},y)\tag{5.1}$$
综上,利用(5.1)给出的初始值,用(4.6)可递推求出序列 $\delta^{(k)}$ ,从而根据(4.1)求出梯度 $\nabla_{W^{(k)}}J$ 和 $\nabla_{b^{(k)}}J$
