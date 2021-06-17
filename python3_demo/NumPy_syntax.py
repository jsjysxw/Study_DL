# 在NumPy中维度(dimensions)叫做轴(axes)，轴的个数叫做秩(rank) NumPy的数组类被称作ndarray
# ndarray.ndim   数组轴的个数，在python的世界中，轴的个数被称作秩
# ndarray.shape  数组的维度。例如一个n排m列的矩阵，它的shape属性将是(2,3)
# ndarray.size 数组元素的总个数，等于shape属性中元组元素的乘积。
# ndarray.dtype 描述数组中元素类型的对象，可以通过创造或指定dtype
# ndarray.itemsize 数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsiz属性值为8(=64/8)

from numpy  import *
a = arange(15).reshape(3, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
# print(a.shape)  # (3, 5)
# print(a.ndim)  # 2
# print(a.dtype.name)  # int32
# print(a.itemsize)  # 4
# print(a.size)  # 15
# print(type(a))  # <class 'numpy.ndarray'>

# 创建数组
b = array([2, 3, 4])
# print(b) # [2 3 4]
# print(b.dtype)  # int32
c = array([1.2, 3.5, 5.1])
# print(c.dtype)  # float64

d = array([(1.5, 2, 3), (4, 5, 6)])
# [[1.5 2.  3. ]
#  [4.  5.  6. ]]

c = array( [ [1,2], [3,4] ], dtype=complex )
# [[1.+0.j 2.+0.j]
#  [3.+0.j 4.+0.j]]

c = zeros( (3,4) )
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

c = ones( (2,3), dtype=int16 )
# [[1 1 1]
#  [1 1 1]]

c = empty( (2,3) )
# [[3.33772792e-307 4.22786102e-307 2.78145267e-307]
#  [4.00537061e-307 9.45708167e-308 0.00000000e+000]]

c = arange( 10, 30, 5 ) # [10 15 20 25]
# 其它函数array, zeros, zeros_like, ones, ones_like, \
# empty, empty_like, arange, linspace, rand, randn, fromfunction, fromfile

# 打印数组
a = arange(6)
# [0 1 2 3 4 5]
b = arange(6).reshape(2,3)
# [[0 1 2]
#  [3 4 5]]
c = arange(8).reshape(2,2,2)
# [[[0 1]
#   [2 3]]
#  [[4 5]
#   [6 7]]]

# 基本运算
a = array( [20,30,40,50] )
b = arange( 4 )
# print(b) # [0 1 2 3]
c = a-b
# print(c) # [20 29 38 47]
# print(b**2) # [0 1 4 9]
# print(10*sin(a)) # [ 9.12945251 -9.88031624  7.4511316  -2.62374854]
# print(a<35) # [ True  True False False]

# NumPy中的乘法运算符*指示按元素计算，矩阵乘法可以使用dot函数或创建矩阵对象实现
A = array( [[1,1],[0,1]] )
B = array( [[2,0],[3,4]] )
# print(A*B)                # elementwise product
# [[2 0]
#  [0 4]]
# print(dot(A,B))                   # matrix product
# [[5 4]
#  [3 4]]


a = ones(3, dtype=int32)
b = linspace(0,pi,3)
# print(b) # [0.         1.57079633 3.14159265]
c = a+b
# print(c) # [1.         2.57079633 4.14159265]
d = exp(c*1j)
# print(d) # [ 0.54030231+0.84147098j -0.84147098+0.54030231j -0.54030231-0.84147098j]
# print(d.dtype.name)  # 'complex128' 许多非数组运算，如计算数组所有元素之和，被作为ndarray类的方法实现
a = random.random((2,3))
# [[0.43993792 0.66211985 0.85532721]
#  [0.18073217 0.01970551 0.53815837]]
# print(a.sum()) # 2.695981027099164
# print(a.min()) # 0.019705505760145958
# print(a.max()) # 0.8553272104221965

# 指定axis参数你可以吧运算应用到数组指定的轴上：(3,4)如果是axis=0，则是第一维度运算（3）的那个维度
b = arange(12).reshape(3,4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
# print(b.sum(axis=0)) # [12 15 18 21] sum of each column
# print(b.min(axis=1)) # [0 4 8] min of each row
# print(b.cumsum(axis=1))  # cumulative sum along each row
# [[ 0  1  3  6]
#  [ 4  9 15 22]
#  [ 8 17 27 38]]

# 通用函数(ufunc) NumPy提供常见的数学函数如sin,cos和exp。在NumPy中，这些叫作“通用函数”(ufunc)
B = arange(3)  # [0 1 2]
# print(exp(B)) # [1.         2.71828183 7.3890561 ]
# print(sqrt(B)) # [0.         1.         1.41421356]
C = array([2., -1., 4.])
# print(add(B, C)) # [2. 0. 6.]
# 更多函数all, alltrue, any, apply along axis, argmax, argmin, argsort, average, bincount
# , ceil, clip, conj, conjugate, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor
# , inner, inv, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod
# , re, round, sometrue, sort, std, sum, trace, transpose, var, vdot, vectorize, where

# 索引，切片和迭代
a = arange(10)**3
# [  0   1   8  27  64 125 216 343 512 729]
# print(a[2]) # 8
# print(a[2:5]) # [ 8 27 64]
a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
# print(a) # [-1000     1 -1000    27 -1000   125   216   343   512   729]
# print(a[ : :-1])                                 # reversed a

def f(x,y):
    return 10*x+y
b = fromfunction(f,(5,4),dtype=int)
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]
#  [30 31 32 33]
#  [40 41 42 43]]
# print(b[2,3]) # 23
# print(b[0:5, 1])      # [ 1 11 21 31 41]     # each row in the second column of b
# print(b[ : ,1])        # [ 1 11 21 31 41]     # equivalent to the previous example
# print(b[1:3, : ])    # each column in the second and third row of b
# [[10 11 12 13]
#  [20 21 22 23]]
# print(b[-1]) # the last row. Equivalent to b[-1,:] 少于轴数的索引被提供时，确失的索引被认为是整个切片：
# [40 41 42 43]

# x[1,2,…] 等同于 x[1,2,:,:,:],
# x[…,3] 等同于 x[:,:,:,:,3]
# x[4,…,5,:] 等同 x[4,:,:,5,:].

# 多维数组取第一维度for row in b:
# for row in b:
#     print(row)
# 想对每个数组中元素进行运算，我们可以使用flat属性  for element in b.flat:

a = floor(10*random.random((3,4)))
# [[7. 4. 8. 5.]
#  [1. 1. 0. 7.]
#  [0. 6. 2. 9.]]
# print(a.ravel())   # [7. 4. 8. 5. 1. 1. 0. 7. 0. 6. 2. 9.]
a.shape = (6, 2)
# print(a.transpose()) #转置
# [[7. 8. 1. 0. 0. 2.]
#  [4. 5. 1. 7. 6. 9.]]
# reshape函数改变参数形状并返回它，而resize函数改变数组自身。

# 组合(stack)不同的数组
a = floor(10*random.random((2,2)))
# [[7. 9.]
#  [9. 8.]]
b = floor(10*random.random((2,2)))
# [[9. 5.]
#  [6. 7.]]
# print(vstack((a,b)))
# [[7. 9.]
#  [9. 8.]
#  [9. 5.]
#  [6. 7.]]
# print(hstack((a,b)))
# [[7. 9. 9. 5.]
#  [9. 8. 6. 7.]]
# 函数column_stack以列将一维数组合成二维数组，它等同与vstack对一维数组。
# 维度比二维更高的数组，hstack沿着第二个轴组合，vstack沿着第一个轴组合,concatenate允许可选参数给出组合时沿着的轴

# r_[]和c_[]对创建沿着一个方向组合的数很有用，它们允许范围符号(“:”):
# print( r_[1:4,0,4])  #[1 2 3 0 4]

# 将一个数组分割(split)成几个小数组
# 使用hsplit你能将数组沿着它的水平轴分割，或者指定返回相同形状数组的个数，或者指定在哪些列后发生分割
# vsplit沿着纵向的轴分割，array split允许指定沿哪个轴分割。

# 复制和视图
# 完全不拷贝
a = arange(12)
b = a            # no new object is created
# print(b is a)    #True       # a and b are two names for the same ndarray object
b.shape = 3,4    # changes the shape of a
# print(a.shape)   # (3, 4)

# 视图(view)和浅复制
# 不同的数组对象分享同一个数据。视图方法创造一个新的数组对象指向同一数据。
c = a.view()
# print(c is a)  # False
# print(c.base is a)         # True      # c is a view of the data owned by a
# print(c.flags.owndata)   # False
c.shape = 2,6                      # a's shape doesn't change
# print(a.shape)   # (3, 4)
c[0,4] = 1234                      # a's data changes
# print(a)
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]

# 切片数组返回它的一个视图：
s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
# print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]

# 深复制
# 这个复制方法完全复制数组和它的数据。
d = a.copy()                          # a new array object with new data is created
# print(d is a)  #  False

# print(d.base is a)     #  False                     # d doesn't share anything with a
d[0,0] = 9999
# print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]


# 函数和方法(method)总览
# 创建数组
# arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace,
# logspace, mgrid, ogrid, ones, ones_like, r , zeros, zeros_like

# 转化
# astype, atleast 1d, atleast 2d, atleast 3d, mat

# 操作
# array split, column stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, item,
# newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

# 询问
# all, any, nonzero, where

# 排序
# argmax, argmin, argsort, max, min, ptp, searchsorted, sort

# 运算
# choose, compress, cumprod, cumsum, inner, fill, imag, prod, put, putmask, real, sum

# 基本统计
# cov, mean, std, var

# 基本线性代数
# cross, dot, outer, svd, vdot

# 广播法则(rule)
# 广播法则能使通用函数有意义地处理不具有相同形状的输入。
#
# 广播第一法则是，如果所有的输入数组维度不都相同，一个“1”将被重复地添加在维度较小
# 的数组上直至所有的数组拥有一样的维度。
#
# 广播第二法则确定长度为1的数组沿着特殊的方向表现地好像它有沿着那个方向最大形状的大小。
# 对数组来说，沿着那个维度的数组元素的值理应相同。
#
# 应用广播法则之后，所有数组的大小必须匹配。


import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainload = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testload = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



if __name__ == '__main__':
    dataiter = iter(trainload)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print('  '.join('%5s' % classes[labels[j]] for j in range(4)))



