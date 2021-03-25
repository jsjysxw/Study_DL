import matplotlib.pyplot as plt
import numpy as np

# 从0开始，相当于x=【0，1，2，3】 y=【1，2，3，4】

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

# x=【1，2，3，4】 y=【1，4，9，16】  'ro' 红色圆圈

# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()


# 0到5之间每隔0.2取一个数

# t = np.arange(0., 5., 0.2)
# # 红色的破折号，蓝色的方块，绿色的三角形
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()



# subplot()中的参数分别指定了numrows、numcols、fignum，其中fignum的取值范围为1到numrows*numcols
# 其实subplot中的参数【111】本应写作【1,1,1】，但是如果这三个参数都小于10(其实就是第三个参数小于10)就可以省略逗号
# 如果你想手动放置一个axes，也就是它不再是一个矩形方格，你就可以使用命令axes()，它可以让坐标系位于任何位置，axes([left,bottom,width,height])
# ，其中所有的值都是0到1(axes([0.3,0.4,0.2,0.3])表示的是该坐标系位于figure的(0.3,0.4)处，其宽度和长度分别为figure横坐标和纵坐标总长的0.2和0.3)

# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
#
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
#
# plt.figure("2subplot", figsize=(6, 6))
# plt.subplot(221)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#
# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()


# subplot和axes的区别就在于axes大小和位置更加随意。
# 你可以创建多个figure，通过调用figure()，其参数为figure的编号。当然每个figure可以包含多个subplot或者是多个axes

# plt.figure(1)                # 编号为1的figure
# plt.subplot(211)             # figure1中的第一个子图
# plt.plot([1, 2, 3])
# plt.subplot(212)             # figure1中的第二个子图
# plt.plot([4, 5, 6])
#
# plt.figure(2)                # figure2
# plt.plot([4, 5, 6])          # 默认使用subplot(111),此时figure2为当
#                              # 前figure
# plt.figure(1)                # 设置figure1为当前figure;
#                              # 但是subplot(212)为当前子图
# plt.subplot(211)             # 使subplot(211)为当前子图
# plt.title('Easy as 1, 2, 3') # 对subplot(211)命名
# plt.show()



#直方图  text()命令可以被用来在任何位置添加文字，xlabel()、ylabel()、title()被用来在指定位置添加文字。

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
#
# # 直方图 50为显示50个柱子
# n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75, stacked=True)
#
# plt.xlabel('Smarts')
# # 所有text()命令返回一个matplotlib.text.Text实例，像上面的线一样，可以通过关键字参数在text()定制文本样式，也可以通过setp()来定制文字的样式：
# # t = plt.xlabel('my data', fontsize=14, color='red')
# # setp(t,color='blue')
#
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()



# 注释文本 ：使用text()命令可以在Axes中任意位置放置文本，一个普遍的文本用法是对一些特性进行注释，annotate()方法让添加注释变得很容易。
# 对于注释有两点需要注意：需要被注释的地方，使用xy参数来指出，还有就是注释文本所放置的位置，使用参数xytext来指定位置

# ax = plt.subplot(111)
#
# t = np.arange(0.0, 5.0, 0.01)
# s = np.cos(2*np.pi*t)
# line, = plt.plot(t, s, lw=2)
#
# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
#
# plt.ylim(-2,2)
# plt.show()



# 下面是一个例子，对于同样的数据，在Y轴使用不同刻度下的曲线图：
# 注意是：Y轴使用不同刻度
# 在区间[0,1]制造一些数据

# np.random.normal为高斯分布
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))


# 创建一个窗口

plt.figure(1)


# 线性

plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)


# 对数

plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log

plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.05)
plt.title('symlog')
plt.grid(True)


# logit

plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

plt.show()