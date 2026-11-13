# B04：写 nn.Module 之前，先补 Python 的类语法
#
# 你懂 C++ 的类和对象，这一课主要是"翻译"：
#   C++ 的 this          ->  Python 的 self
#   C++ 的构造函数        ->  Python 的 __init__
#   C++ 的 class A : B   ->  Python 的 class A(B)
#   C++ 的 public: 成员   ->  Python 的 self.xxx = ...（必须显式写！）
#
# 两个 Python 特有的坑，后面写 nn.Module 时天天会遇到


# ============================================================
# 一、最小的类：一个"手写线性层"
# ============================================================
class TinyLinear:
    def __init__(self, w1, w2, b):
        # __init__ 就是构造函数，创建对象时自动调用
        # 参数 self 就是"正在被创建的那个对象"，相当于 C++ 的 this
        self.w1 = w1
        self.w2 = w2
        self.b = b
        # ★ 必须写 self.xxx 才会变成成员变量
        #   如果只写 w1 = w1，那就是个临时变量，函数一结束就没了

    def forward(self, x1, x2):
        # 普通成员函数：第一个参数永远是 self，但调用时不用传
        return self.w1 * x1 + self.w2 * x2 + self.b


m = TinyLinear(0.3, 0.5, 0.1)      # 创建对象，__init__ 自动执行
print("m.forward(1, 2) =", m.forward(1, 2))
print("m.w1 =", m.w1)
print("m.b  =", m.b)
print("m 的类型 =", type(m).__name__)

# 注意 m.forward(1, 2) 只传了两个参数
# self 是自动传进去的（Python 帮你把 m 放到第一个位置）


# ============================================================
# 二、坑 1：局部变量不会变成成员变量
# ============================================================
class Bad1:
    def __init__(self):
        w = 1.0            # 忘了写 self.，这只是个临时变量

    def f(self):
        return w           # 这里找不到 w


class Bad2:
    def __init__(self):
        pass               # 什么都没存

    def f(self):
        return self.w      # 从来没赋过值


for cls in (Bad1, Bad2):
    cls_name = cls.__name__
    try:
        print(cls().f())
    except Exception as e:
        print(cls_name, "->", type(e).__name__, ":", e)

# 输出：
# Bad1 -> NameError : name 'w' is not defined
# Bad2 -> AttributeError : 'Bad2' object has no attribute 'w'
#
# ★ 两个错误信息不一样，但病因同一个：成员变量没建起来
#   NameError      : 完全找不到这个名字（连 self 都没写）
#   AttributeError : 对象上没这个属性（写了 self.w，但没赋值）
#
# 后面写 nn.Module 时，最常见的报错就是 AttributeError: 'Net' object has no attribute 'fc'
# 原因永远是：__init__ 里漏了 self.fc = ...


# ============================================================
# 三、继承和 super()
# ============================================================
class Base:
    def __init__(self):
        self.name = "base"

    def hello(self):
        return "hello from " + self.name


class Child(Base):                 # 括号里写父类，相当于 C++ 的 class Child : public Base
    def __init__(self):
        super().__init__()         # 先让父类把自己的初始化做完 # super()就是父类的意思
        self.name = "child"        # 再覆盖

    def hello(self):
        return super().hello() + " (via child)"


c = Child()
print(c.hello())

# ★ super().__init__() 这一行绝对不能省
#   父类可能有一堆初始化工作（nn.Module 就要注册参数表、hook 等）
#   不调用它，后面的功能全部失灵，而且报错信息往往看不懂


# ============================================================
# 四、坑 2：写在 class 层级的可变对象，被所有实例共享
# ============================================================
class Shared:
    bag = []                  # 写在 class 里，不是 __init__ 里

    def add(self, v):
        self.bag.append(v)


a = Shared()
b = Shared()
a.add(1)
b.add(2)
print("a.bag =", a.bag, " b.bag =", b.bag)
# 输出：a.bag = [1, 2]  b.bag = [1, 2]
# a 和 b 竟然共享同一个列表！

# 原因：bag 是"类属性"，属于类本身，所有实例公用一份（类似 C++ 的 static）
# 数字、字符串这类不可变对象看不出来；
# 但列表、字典、张量这种可变对象，改一个就是改全体

# ★ 这条对 nn.Module 极其关键：
#   层必须写在 __init__ 里（self.fc = nn.Linear(...)），不能写在 class 层级
#   否则所有模型实例共用同一个层，训练两个模型会互相干扰


# ============================================================
# 五、字典和 for 遍历（后面打印参数要用）
# ============================================================
params = {"fc.weight": [1, 2], "fc.bias": [1]}

def count_elems(shape_list):
    total = 1
    for s in shape_list:
        total *= s
    return total

for name, shape in params.items():        # .items() 同时取出键和值
    print(f"{name:12} shape={shape}  元素数={count_elems(shape)}")

# 这就是后面 named_parameters() 的用法雏形：
#   它返回的就是一个"名字 -> 张量"的迭代器
#   for name, p in model.named_parameters(): print(name, p.shape)


# ============================================================
# 总结：C++ 和 Python 对照
# ============================================================
# 概念           C++                        Python
# ---------------------------------------------------------------
# 定义类         class A { }                class A:
# 构造函数       A() { }                    def __init__(self):
# this 指针      this->x                     self.x
# 成员变量       int x;（声明就能用）         必须 self.x = ... 显式创建
# 继承           class B : public A          class B(A):
# 调父类构造     A::A()                      super().__init__()
# 静态成员       static int x;              写在 class 层级（但可变对象要小心）
# 私有           private:                   _ 前缀约定（语言层不强制）
