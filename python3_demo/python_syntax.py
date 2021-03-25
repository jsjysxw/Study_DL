import keyword
# 输出当前版本的所有关键字
print(keyword.kwlist)

str = 'Runoob'
print(str[0:-1])  # 输出第一个到倒数第二个的所有字符
print(str[2:5])  # 输出从第三个开始到第五个的字符
print(str[2:])  # 输出从第三个开始后的所有字符
print(str[1:5:2])  # 输出从第二个开始到第五个且每隔两个的字符
print(str * 2)  # 输出字符串两次
print(str + '你好')  # 连接字符串
print('hello\nrunoob')  # 使用反斜杠(\)+n转义特殊字符
print(r'hello\nrunoob')  # 在字符串前面添加一个 r，表示原始字符串，不会发生转义

# 内置的 type() 函数可以用来查询变量所指的对象类型
# isinstance(a, int)  isinstance 来判断类型
# 元组（tuple）与列表类似，不同之处在于元组的元素不能修改
# a ** b    为a的b次方
# >>> 9//2 = 4   # >>> -9//2 = -5  取整除 - 向下取接近商的整数

# 字符串，列表或元组对象都可用于创建迭代器：
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
print (next(it))   # 输出迭代器的下一个元素
print (next(it))

# 迭代器对象可以使用常规for语句进行遍历：
for x in it:
    print (x, end=" ")   # 3 4

# 正常没有发生异常则执行 else 部分的语句
try:
    print("111")
    # runoob()
except AssertionError as error:
    print(error)
else:
    try:
        with open('file.log') as file:
            read_data = file.read()
    except FileNotFoundError as fnf_error:
        print(fnf_error)
finally:
    print('这句话，无论异常是否发生都会执行。')

# 以下实例如果 x 大于 5 就触发异常:
x = 10
if x > 5:
    raise Exception('x 不能大于 5。x 的值为: {}'.format(x))