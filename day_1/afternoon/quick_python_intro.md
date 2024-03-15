---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python
  language: python3
  name: python3
---



# A Quick Introduction to Python

This notebook provides a super quick introduction to Python.

We assume that participants needing a slower treatment will review the
first few [QuantEcon Python programming
lectures](https://python-programming.quantecon.org/intro.html) in their own
time.

+++

## Example Task: Plotting a White Noise Process

Task: simulate and plot the white noise
process $ \epsilon_0, \epsilon_1, \ldots, \epsilon_T $, where each draw $ \epsilon_t $ is independent standard normal.


### Version 1

Here are a few lines of code that perform the task we set

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt   

ϵ_values = np.random.randn(100)   # 100 draws from N(0, 1)
plt.plot(ϵ_values)                # Plot draws
plt.show()
```

Let’s discuss some aspects of this program.

+++

#### Imports

The first two lines of the program import functionality from external code
libraries.

The first line imports [NumPy](https://python-programming.quantecon.org/numpy.html), a favorite Python package for tasks like

- working with arrays (vectors and matrices)  
- common mathematical functions like `cos` and `sqrt`  
- generating random numbers  
- linear algebra, etc.  


After `import numpy as np` we have access to these attributes via the syntax `np.attribute`.

Here’s two more examples

```{code-cell}
np.sqrt(4)
```

```{code-cell}
np.log(4)
```

#### Why So Many Imports?

The reason is that the core language is deliberately kept small, so that it’s easy to learn, maintain and improve.

When you want to do something interesting with Python, you almost always need
to import additional functionality.

+++

#### Importing Names Directly

Recall this code that we saw above

```{code-cell}
import numpy as np
np.sqrt(4)
```

Here’s another way to access NumPy’s square root function

```{code-cell}
from numpy import sqrt
sqrt(4)
```

This is also fine -- simple statements typing vs less explicit

+++



### A Version with a For Loop

Here’s a (less efficient) version that illustrates `for` loops and Python lists.


```{code-cell}
ts_length = 100
ϵ_values = []       # Empty list

for i in range(ts_length):
    e = np.random.randn()
    ϵ_values.append(e)

plt.plot(ϵ_values)
plt.show()
```

How does it work?

How do you like significant whitespace??

+++

#### Lists


Consider the statement `ϵ_values = []`, which creates an empty list.

Lists are a native Python data structure used to group a collection of objects.

Items in lists are ordered, and duplicates are allowed in lists.

For example, try

```{code-cell}
x = [10, 'foo', False]
type(x)
```

When adding a value to a list, we can use the syntax `list_name.append(some_value)`


```{code-cell}
x.append(2.5)
x
```

Here `append()` is what’s called a **method**, which is a function "attached to" an object -- in this case, the list `x`.


- Python objects such as lists, strings, etc. all have methods that are used to manipulate data contained in the object.  
- String objects have string methods, list objects have list methods, etc.

Another useful list method is `pop()`

```{code-cell}
x
```

```{code-cell}
x.pop()
```

```{code-cell}
x
```

Lists in Python are zero-based (as in C, Java or Go), so the first element is referenced by `x[0]`

```{code-cell}
x[0]   # First element of x
```

```{code-cell}
x[1]   # Second element of x
```



### While Loops


For the purpose of illustration, let’s modify our program to use a `while` loop instead of a `for` loop.


```{code-cell}
ts_length = 100
ϵ_values = []
i = 0
while i < ts_length:
    e = np.random.randn()
    ϵ_values.append(e)
    i = i + 1             # Equivalent: i += 1
plt.plot(ϵ_values)
plt.show()
```

How does it work?

+++

**Exercise**

Plot the balance of a bank account over $0, \ldots, T$ when $T=50$.

* There are no withdraws 
* The initial balance is $ b_0 = 10 $ and the interest rate is $ r = 0.025$.

The balance updates from period $ t $ to $ t+1 $ according to $ b_{t+1} = (1 + r) b_t $.

Your task is to generate and plot the sequence $b_0, b_1, \ldots, b_T $.

You can use a Python list to store this sequence, or a NumPy array.

In the second case, you can use a statement such as

```{code-cell}
b = np.empty(T+1)   # Allocate memory to store all b_t
```

and then populating `b[t]` in a for loop.


```{code-cell}
for i in range(12):
    print("Solution below.")
```



**Solution**


```{code-cell}
r = 0.025         # interest rate
T = 50            # end date
b = np.empty(T+1) # an empty NumPy array, to store all b_t
b[0] = 10         # initial balance

for t in range(T):
    b[t+1] = (1 + r) * b[t]

plt.plot(b, label='bank balance')
plt.legend()
plt.show()
```


**Exercise**

Simulate and plot the correlated time series

$$
    x_{t+1} = \alpha \, x_t + \epsilon_{t+1}
    \quad \text{where} \quad
    x_0 = 0
    \quad \text{and} \quad t = 0,\ldots,T
$$

were $ \{\epsilon_t\} $ is IID and standard normal.

In your solution, restrict your import statements to

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
```

Set $ T=200 $ and $ \alpha = 0.9 $.



```{code-cell}
for i in range(12):
    print("Solution below.")
```



**Solution**


Here’s one solution.

```{code-cell}
α = 0.9
T = 200
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    x[t+1] = α * x[t] + np.random.randn()

plt.plot(x)
plt.show()
```

**Exercise** 

Plot three simulated time series,
one for each of the cases $ \alpha=0 $, $ \alpha=0.8 $ and $ \alpha=0.98 $.

Use a `for` loop to step through the $ \alpha $ values.

If you can, add a legend, to help distinguish between the three time series.

- If you call the `plot()` function multiple times before calling `show()`, all of the lines you produce will end up on the same figure.  
- For the legend, noted that suppose `var = 42`, the expression `f'foo{var}'` evaluates to `'foo42'`.  


```{code-cell}
for i in range(12):
    print("Solution below.")
```



**Solution**

```{code-cell}
α_values = [0.0, 0.8, 0.98]
T = 200
x = np.empty(T+1)

for α in α_values:
    x[0] = 0
    for t in range(T):
        x[t+1] = α * x[t] + np.random.randn()
    plt.plot(x, label=f'$\\alpha = {α}$')

plt.legend()
plt.show()
```


## Flow Control

One important aspect of essentially all programming languages is branching and
conditions.

In Python, conditions are usually implemented with if-else syntax.

Here’s an example, that prints -1 for each negative number in an array and 1
for each nonnegative number

```{code-cell}
numbers = [-9, 2.3, -11, 0]
```

```{code-cell}
for x in numbers:
    if x < 0:
        print(-1)
    else:
        print(1)
```

Now, write a new solution to Exercise 3 that does not use an existing function
to compute the absolute value.

Replace this existing function with an if-else condition.


```{code-cell}
for i in range(12):
    print("Solution below.")
```

**Solution**

Here’s one way:

```{code-cell}
α = 0.9
T = 200
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    if x[t] < 0:
        abs_x = - x[t]
    else:
        abs_x = x[t]
    x[t+1] = α * abs_x + np.random.randn()

plt.plot(x)
plt.show()
```

Here’s a shorter way to write the same thing:

```{code-cell}
α = 0.9
T = 200
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    abs_x = - x[t] if x[t] < 0 else x[t]
    x[t+1] = α * abs_x + np.random.randn()

plt.plot(x)
plt.show()
```

## Data Types


Computer programs typically keep track of a range of data types.

For example, `1.5` is a floating point number, while `1` is an integer.

One simple data type is **Boolean values**, which can be either `True` or `False`

```{code-cell}
x = True
x
```

We can check the type of any object in memory using the `type()` function.

```{code-cell}
type(x)
```

In the next line of code, the interpreter evaluates the expression on the right of = and binds y to this value

```{code-cell}
y = 100 < 10
y
```

```{code-cell}
type(y)
```

In arithmetic expressions, `True` is converted to `1` and `False` is converted `0`.

This is called **Boolean arithmetic** and is often useful in programming.

Here are some examples

```{code-cell}
x + y
```

```{code-cell}
x * y
```


```{code-cell}
bools = [True, True, False, True]  # List of Boolean values
sum(bools)
```


### Containers

Python has several basic types for storing collections of (possibly heterogeneous) data.

We have already discussed lists.

A related data type is **tuples**, which are "immutable" lists

```{code-cell}
x = ('a', 'b')  # Parentheses instead of the square brackets
x = 'a', 'b'    # Or no brackets --- the meaning is identical
x
```

```{code-cell}
type(x)
```

In Python, an object is called **immutable** if, once created, the object cannot be changed.

Conversely, an object is **mutable** if it can still be altered after creation.

Python lists are mutable

```{code-cell}
x = [1, 2]
x[0] = 10
x
```

But tuples are not

```{code-cell}
x = (1, 2)
x[0] = 10
```

Tuples (and lists) can be “unpacked” as follows

```{code-cell}
integers = (10, 20, 30)
x, y, z = integers
x
```

```{code-cell}
y
```

You’ve actually [seen an example of this](https://python-programming.quantecon.org/about_py.html#tuple-unpacking-example) already.

Tuple unpacking is convenient and we’ll use it often.

+++

#### Slice Notation


To access multiple elements of a sequence (a list, a tuple or a string), you can use Python’s slice
notation.

For example,

```{code-cell}
a = ["a", "b", "c", "d", "e"]
a[1:]
```

```{code-cell}
a[1:3]
```

The rule is `a[m:n]` returns `n - m` elements, starting at `a[m]`.

Also:

```{code-cell}
a[-2:]  # Last two elements of the list
```
```{code-cell}
s = 'foobar'
s[-3:]  # Last three elements
```


## Iterating


One of the most important tasks in computing is stepping through a
sequence of data and performing a given action.

One of Python’s strengths is its simple, flexible interface to iteration.

### Looping over Different Objects

Many Python objects are "iterable", in the sense that they can be looped over.

To give an example, let’s write the file us_cities.txt, which lists US cities and their population, to the present working directory.



```{code-cell}
%%writefile us_cities.txt
new york: 8244910
los angeles: 3819702
chicago: 2707120
houston: 2145146
philadelphia: 1536471
phoenix: 1469471
san antonio: 1359758
san diego: 1326179
dallas: 1223229
```

Suppose that we want to make the information more readable, by capitalizing names and adding commas to mark thousands.

The program below reads the data in and makes the conversion:

```{code-cell}
data_file = open('us_cities.txt', 'r')
for line in data_file:
    city, population = line.split(':')         # Tuple unpacking
    city = city.title()                        # Capitalize city names
    population = f'{int(population):,}'        # Add commas to numbers
    print(city.ljust(15) + population)
data_file.close()
```


### Looping without Indices

Python tends to favor looping without explicit indexing.

For example,

```{code-cell}
x_values = [1, 2, 3]  # Some iterable x
for x in x_values:
    print(x * x)
```

is preferred to

```{code-cell}
for i in range(len(x_values)):
    print(x_values[i] * x_values[i])
```

Python provides some facilities to simplify looping without indices.

One is `zip()`, which is used for stepping through pairs from two sequences.

For example, try running the following code

```{code-cell}
countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')
for country, city in zip(countries, cities):
    print(f'The capital of {country} is {city}')
```

If we actually need the index from a list, one option is to use `enumerate()`.

To understand what `enumerate()` does, consider the following example

```{code-cell}
letter_list = ['a', 'b', 'c']
for index, letter in enumerate(letter_list):
    print(f"letter_list[{index}] = '{letter}'")
```



### List Comprehensions

[List comprehensions](https://en.wikipedia.org/wiki/List_comprehension) are an elegant Python tool for creating lists.

Consider the following example, where the list comprehension is on the
right-hand side of the second line

```{code-cell}
animals = ['dog', 'cat', 'bird']
plurals = [animal + 's' for animal in animals]
plurals
```

Here’s another example

```{code-cell}
range(8)
```

```{code-cell}
doubles = [2 * x for x in range(8)]
doubles
```

## Comparisons and Logical Operators


### Comparisons

In Python we can chain inequalities

```{code-cell}
1 < 2 < 3
```

```{code-cell}
1 <= 2 <= 3
```

When testing for equality we use `==`

```{code-cell}
x = 1    # Assignment
x == 2   # Comparison
```

For “not equal” use `!=`

```{code-cell}
1 != 2
```

### Combining Expressions

We can combine expressions using `and`, `or` and `not`.

These are the standard logical connectives (conjunction, disjunction and denial)

```{code-cell}
1 < 2 and 'f' in 'foo'
```

```{code-cell}
1 < 2 and 'g' in 'foo'
```

```{code-cell}
1 < 2 or 'g' in 'foo'
```

```{code-cell}
not not True
```


## Coding Style and Documentation

A consistent coding style make code easier to understand and maintain.

You can find Python programming philosophy by typing `import this` at the prompt.

See also the Python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/).



+++


**Exercise**

Part 1: Given two numeric lists or tuples `x_vals` and `y_vals` of equal length, compute
their inner product using `zip()`.

Part 2: In one line, count the number of even numbers in 0,…,99.

Part 3: Given `pairs = ((2, 5), (4, 2), (9, 8), (12, 10))`, count the number of pairs `(a, b)`
such that both `a` and `b` are even.

Hint: `x % 2` returns 0 if `x` is even, 1 otherwise.

Part 4: Write a function that takes a string as an argument and returns the number of capital letters in the string.

`'foo'.upper()` returns `'FOO'`.


```{code-cell}
for i in range(12):
    print("Solution below.")
```


**Part 1 Solution:**

Here’s one possible solution

```{code-cell}
x_vals = [1, 2, 3]
y_vals = [1, 1, 1]
sum([x * y for x, y in zip(x_vals, y_vals)])
```

This also works

```{code-cell}
sum(x * y for x, y in zip(x_vals, y_vals))
```

**Part 2 Solution:**

One solution is

```{code-cell}
sum([x % 2 == 0 for x in range(100)])
```

This also works:

```{code-cell}
sum(x % 2 == 0 for x in range(100))
```

Some less natural alternatives that nonetheless help to illustrate the
flexibility of list comprehensions are

```{code-cell}
len([x for x in range(100) if x % 2 == 0])
```

and

```{code-cell}
sum([1 for x in range(100) if x % 2 == 0])
```

**Part 3 Solution:**

Here’s one possibility

```{code-cell}
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs])
```


+++

**Part 4 Solution:**

Here’s one solution:

```{code-cell}
def f(string):
    count = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            count += 1
    return count

f('The Rain in Spain')
```

Alternatively, 

```{code-cell}
def count_uppercase_chars(s):
    return sum([c.isupper() for c in s])

count_uppercase_chars('The Rain in Spain')
```


+++



## Defining Functions

### Basic Syntax

Here’s a very simple Python function

```{code-cell}
def f(x):
    return 2 * x + 1
```

Now that we’ve defined this function, let’s *call* it and check whether it does what we expect:

```{code-cell}
f(1)   
```

```{code-cell}

f(10)
```

Here’s a longer function, that computes the absolute value of a given number.

(Such a function already exists as a built-in, but let’s write our own for the
exercise.)

```{code-cell}
def new_abs_function(x):
    if x < 0:
        abs_value = -x
    else:
        abs_value = x
    return abs_value
```

Let’s call it to check that it works:

```{code-cell}
print(new_abs_function(3))
print(new_abs_function(-3))
```

Note that a function can have arbitrarily many `return` statements (including zero).

Functions without a return statement automatically return the special Python object `None`.



+++

### Keyword Arguments


The following example illustrates the syntax

```{code-cell}
def f(x, a=1, b=1):
    return a + b * x
```

The keyword argument values we supplied in the definition of `f` become the default values

```{code-cell}
f(2)
```

They can be modified as follows

```{code-cell}
f(2, a=4, b=5)
```


### The Flexibility of Python Functions


- Any number of functions can be defined in a given file.  
- Functions can be (and often are) defined inside other functions.  
- Any object can be passed to a function as an argument, including other functions.  
- A function can return any kind of object, including functions.  

+++


### One-Line Functions: `lambda`


The `lambda` keyword is used to create simple functions on one line.

For example, 

```{code-cell}
def f(x):
    return x**3
```

is equivalent to.

```{code-cell}
f = lambda x: x**3
```

One use case is "anonymous" functions


```{code-cell}
from scipy.integrate import quad
quad(lambda x: x**3, 0, 2)
```


+++

### Random Draws

Consider again the code 

```{code-cell}
ts_length = 100
ϵ_values = []   # empty list

for i in range(ts_length):
    e = np.random.randn()
    ϵ_values.append(e)

plt.plot(ϵ_values)
plt.show()
```

We can break this down as follows:


```{code-cell}
def generate_data(n):
    ϵ_values = []
    for i in range(n):
        e = np.random.randn()
        ϵ_values.append(e)
    return ϵ_values

data = generate_data(100)
plt.plot(data)
plt.show()
```


Here's an alternative where we pass a function to a function:


```{code-cell}
def generate_data(n, generator_type):
    ϵ_values = []
    for i in range(n):
        e = generator_type()
        ϵ_values.append(e)
    return ϵ_values

data = generate_data(100, np.random.uniform)
plt.plot(data)
plt.show()
```



**Exercise**

The binomial random variable $Y$ gives the number of successes in $ n $ binary trials, where each trial
succeeds with probability $ p $.

Without any import besides `from numpy.random import uniform`, write a function
`binomial_rv` such that `binomial_rv(n, p)` generates one draw of $ Y $.

Hint: If $ U $ is uniform on $ (0, 1) $ and $ p \in (0,1) $, then the expression `U < p` evaluates to `True` with probability $ p $.

```{code-cell}
for i in range(12):
    print("Solution below.")
```


**Solution** 

Here's one solution:

```{code-cell}
from numpy.random import uniform

def binomial_rv(n, p):
    count = 0
    for i in range(n):
        U = uniform()
        if U < p:
            count = count + 1    # Or count += 1
    return count

binomial_rv(10, 0.5)
```


## OOP: Objects and Methods


The traditional programming paradigm (Fortran, C, MATLAB, etc.) is called procedural.


Another important paradigm is object-oriented programming (OOP) 

In the OOP paradigm, data and functions are bundled together into “objects” — and functions in this context are referred to as **methods**.

Methods are called on to transform the data contained in the object.

- Think of a Python list that contains data and has methods such as `append()` and `pop()` that transform the data.  

A third paradigm is functional programming 

* Built on the idea of composing functions.
* We'll discuss this more when we get to JAX

Python is a pragmatic language that blends object-oriented, functional and procedural styles.

But at a foundational level, Python *is* object-oriented.

By this we mean that, in Python, *everything is an object*.


### Objects


In Python, an *object* is a collection of data and instructions held in computer memory that consists of

1. a type  
1. a unique identity  
1. data (i.e., content, reference count)  
1. methods  



+++

#### Type


Python provides for different types of objects, to accommodate different categories of data.

For example

```{code-cell}
s = 'This is a string'
type(s)
```

```{code-cell}
x = 42   # Now let's create an integer
type(x)
```

The type of an object matters for many expressions.

For example, the addition operator between two strings means concatenation

```{code-cell}
'300' + 'cc'
```

On the other hand, between two numbers it means ordinary addition

```{code-cell}
300 + 400
```

Consider the following expression

```{code-cell}
'300' + 400
```

Here we are mixing types, and it’s unclear to Python whether the user wants to

Python is *strongly typed* -- throws an error rather than trying to perform
hidden type conversion.




+++

#### Identity


In Python, each object has a unique identifier, which helps Python (and us) keep track of the object.

The identity of an object can be obtained via the `id()` function

```{code-cell}
y = 2.5
z = 2.5
id(y)
```

```{code-cell}
id(z)
```

In this example, `y` and `z` happen to have the same value (i.e., `2.5`), but they are not the same object.

The identity of an object is in fact just the address of the object in memory.

+++

#### Object Content: Data and Attributes


If we set `x = 42` then we create an object of type `int` that contains
the data `42`.

In fact, it contains more, as the following example shows

```{code-cell}
x = 42
x
```

```{code-cell}
x.imag
```

```{code-cell}
x.__class__
```

When Python creates this integer object, it stores with it various auxiliary information, such as the imaginary part, and the type.

Any name following a dot is called an *attribute* of the object to the left of the dot.

- e.g.,`imag` and `__class__` are attributes of `x`.  


We see from this example that objects have attributes that contain auxiliary information.

They also have attributes that act like functions, called *methods*.

These attributes are important, so let’s discuss them in-depth.



+++

### Methods


Methods are *functions that are bundled with objects*.

Formally, methods are attributes of objects that are **callable** – i.e., attributes that can be called as functions

```{code-cell}
x = ['foo', 'bar']
callable(x.append)
```

```{code-cell}
callable(x.__doc__)
```

Methods typically act on the data contained in the object they belong to, or combine that data with other data

```{code-cell}
x = ['a', 'b']
x.append('c')
s = 'This is a string'
s.upper()
```

```{code-cell}
s.lower()
```

```{code-cell}
s.replace('This', 'That')
```

A great deal of Python functionality is organized around method calls.

For example, consider the following piece of code

```{code-cell}
x = ['a', 'b']
x[0] = 'aa'  # Item assignment using square bracket notation
x
```

It doesn’t look like there are any methods used here, but in fact the square bracket assignment notation is just a convenient interface to a method call.

What actually happens is that Python calls the `__setitem__` method, as follows

```{code-cell}
x = ['a', 'b']
x.__setitem__(0, 'aa')  # Equivalent to x[0] = 'aa'
x
```

(If you wanted to you could modify the `__setitem__` method, so that square bracket assignment does something totally different)

+++

## Inspection Using Rich

There’s a nice package called [rich](https://github.com/Textualize/rich) that
helps us view the contents of an object.

For example,

```{code-cell}
!pip install rich

from rich import inspect
x = 10
inspect(10)
```

If we want to see the methods as well, we can use

```{code-cell}
inspect(10, methods=True)
```

In fact there are still more methods, as you can see if you execute `inspect(10, all=True)`.

+++


## Names and Namespaces

## Variable Names in Python


Consider the Python statement

```{code-cell}
:hide-output: false

x = 42
```

We now know that when this statement is executed, Python creates an object of
type `int` in your computer’s memory, containing

- the value `42`  
- some associated attributes  


But what is `x` itself?

In Python, `x` is called a **name**, and the statement `x = 42` **binds** the name `x` to the integer object we have just discussed.

Under the hood, this process of binding names to objects is implemented as a dictionary—more about this in a moment.

There is no problem binding two or more names to the one object, regardless of what that object is

```{code-cell}
:hide-output: false

def f(string):      # Create a function called f
    print(string)   # that prints any string it's passed

g = f
id(g) == id(f)
```

```{code-cell}
:hide-output: false

g('test')
```

In the first step, a function object is created, and the name `f` is bound to it.

After binding the name `g` to the same object, we can use it anywhere we would use `f`.

What happens when the number of names bound to an object goes to zero?

Here’s an example of this situation, where the name `x` is first bound to one object and then **rebound** to another

```{code-cell}
:hide-output: false

x = 'foo'
id(x)
x = 'bar'  
id(x)
```

In this case, after we rebind `x` to `'bar'`, no names bound are to the first object `'foo'`.

This is a trigger for `'foo'` to be garbage collected.

In other words, the memory slot that stores that object is deallocated and returned to the operating system.

Garbage collection is actually an active research area in computer science.

You can [read more on garbage collection](https://rushter.com/blog/python-garbage-collector/) if you are interested.

+++

## Namespaces


Recall from the preceding discussion that the statement

```{code-cell}
:hide-output: false

x = 42
```

binds the name `x` to the integer object on the right-hand side.

We also mentioned that this process of binding `x` to the correct object is implemented as a dictionary.

This dictionary is called a namespace.

+++

## Definition

A **namespace** is a symbol table that maps names to objects in memory.

Python uses multiple namespaces, creating them on the fly as necessary.

For example, every time we import a module, Python creates a namespace for that module.

To see this in action, suppose we write a script `mathfoo.py` with a single line

```{code-cell}
:hide-output: false

%%file mathfoo.py
pi = 'foobar'
```

Now we start the Python interpreter and import it

```{code-cell}
:hide-output: false

import mathfoo
```

Next let’s import the `math` module from the standard library

```{code-cell}
:hide-output: false

import math
```

Both of these modules have an attribute called `pi`

```{code-cell}
:hide-output: false

math.pi
```

```{code-cell}
:hide-output: false

mathfoo.pi
```

These two different bindings of `pi` exist in different namespaces, each one implemented as a dictionary.

If you wish, you can look at the dictionary directly, using `module_name.__dict__`.

```{code-cell}
:hide-output: false

import math

math.__dict__.items()
```

```{code-cell}
:hide-output: false

import mathfoo

mathfoo.__dict__
```

As you know, we access elements of the namespace using the dotted attribute notation

```{code-cell}
:hide-output: false

math.pi
```

This is entirely equivalent to `math.__dict__['pi']`

```{code-cell}
:hide-output: false

math.__dict__['pi'] 
```

## Viewing Namespaces

As we saw above, the `math` namespace can be printed by typing `math.__dict__`.

Another way to see its contents is to type `vars(math)`

```{code-cell}
:hide-output: false

vars(math).items()
```

If you just want to see the names, you can type

```{code-cell}
:hide-output: false

# Show the first 10 names
dir(math)[0:10]
```

Notice the special names `__doc__` and `__name__`.

These are initialized in the namespace when any module is imported

- `__doc__` is the doc string of the module  
- `__name__` is the name of the module  

```{code-cell}
:hide-output: false

print(math.__doc__)
```

```{code-cell}
:hide-output: false

math.__name__
```

## Interactive Sessions


In Python, **all** code executed by the interpreter runs in some module.

What about commands typed at the prompt?

These are also regarded as being executed within a module — in this case, a module called `__main__`.

To check this, we can look at the current module name via the value of `__name__` given at the prompt

```{code-cell}
:hide-output: false

print(__name__)
```

When we run a script using IPython’s `run` command, the contents of the file are executed as part of `__main__` too.

To see this, let’s create a file `mod.py` that prints its own `__name__` attribute

```{code-cell}
:hide-output: false

%%file mod.py
print(__name__)
```

Now let’s look at two different ways of running it in IPython

```{code-cell}
:hide-output: false

import mod  # Standard import
```

```{code-cell}
:hide-output: false

%run mod.py  # Run interactively
```

In the second case, the code is executed as part of `__main__`, so `__name__` is equal to `__main__`.

To see the contents of the namespace of `__main__` we use `vars()` rather than `vars(__main__)`.

If you do this in IPython, you will see a whole lot of variables that IPython
needs, and has initialized when you started up your session.

If you prefer to see only the variables you have initialized, use `%whos`

```{code-cell}
:hide-output: false

x = 2
y = 3

import numpy as np

%whos
```

## The Global Namespace


Python documentation often makes reference to the “global namespace”.

The global namespace is *the namespace of the module currently being executed*.

For example, suppose that we start the interpreter and begin making assignments.

We are now working in the module `__main__`, and hence the namespace for `__main__` is the global namespace.

Next, we import a module called `amodule`

+++ {"hide-output": false}

```python3
import amodule
```

+++

At this point, the interpreter creates a namespace for the module `amodule` and starts executing commands in the module.

While this occurs, the namespace `amodule.__dict__` is the global namespace.

Once execution of the module finishes, the interpreter returns to the module from where the import statement was made.

In this case it’s `__main__`, so the namespace of `__main__` again becomes the global namespace.

+++

## Local Namespaces


Important fact: When we call a function, the interpreter creates a *local namespace* for that function, and registers the variables in that namespace.

The reason for this will be explained in just a moment.

Variables in the local namespace are called *local variables*.

After the function returns, the namespace is deallocated and lost.

While the function is executing, we can view the contents of the local namespace with `locals()`.

For example, consider

```{code-cell}
:hide-output: false

def f(x):
    a = 2
    print(locals())
    return a * x
```

Now let’s call the function

```{code-cell}
:hide-output: false

f(1)
```

You can see the local namespace of `f` before it is destroyed.

+++

## The `__builtins__` Namespace


We have been using various built-in functions, such as `max(), dir(), str(), list(), len(), range(), type()`, etc.

How does access to these names work?

- These definitions are stored in a module called `__builtin__`.  
- They have their own namespace called `__builtins__`.  

```{code-cell}
:hide-output: false

# Show the first 10 names in `__main__`
dir()[0:10]
```

```{code-cell}
:hide-output: false

# Show the first 10 names in `__builtins__`
dir(__builtins__)[0:10]
```

We can access elements of the namespace as follows

```{code-cell}
:hide-output: false

__builtins__.max
```

But `__builtins__` is special, because we can always access them directly as well

```{code-cell}
:hide-output: false

max
```

```{code-cell}
:hide-output: false

__builtins__.max == max
```

The next section explains how this works …

+++

## Name Resolution


Namespaces are great because they help us organize variable names.

(Type `import this` at the prompt and look at the last item that’s printed)

However, we do need to understand how the Python interpreter works with multiple namespaces.

Understanding the flow of execution will help us to check which variables are in scope and how to operate on them when writing and debugging programs.

At any point of execution, there are in fact at least two namespaces that can be accessed directly.

(“Accessed directly” means without using a dot, as in  `pi` rather than `math.pi`)

These namespaces are

- The global namespace (of the module being executed)  
- The builtin namespace  


If the interpreter is executing a function, then the directly accessible namespaces are

- The local namespace of the function  
- The global namespace (of the module being executed)  
- The builtin namespace  


Sometimes functions are defined within other functions, like so

```{code-cell}
:hide-output: false

def f():
    a = 2
    def g():
        b = 4
        print(a * b)
    g()
```

Here `f` is the *enclosing function* for `g`, and each function gets its
own namespaces.

Now we can give the rule for how namespace resolution works:

The order in which the interpreter searches for names is

1. the local namespace (if it exists)  
1. the hierarchy of enclosing namespaces (if they exist)  
1. the global namespace  
1. the builtin namespace  


If the name is not in any of these namespaces, the interpreter raises a `NameError`.

This is called the **LEGB rule** (local, enclosing, global, builtin).

Here’s an example that helps to illustrate.

Visualizations here are created by [nbtutor](https://github.com/lgpage/nbtutor) in a Jupyter notebook.

They can help you better understand your program when you are learning a new language.

Consider a script `test.py` that looks as follows

```{code-cell}
:hide-output: false

%%file test.py
def g(x):
    a = 1
    x = x + a
    return x

a = 0
y = g(10)
print("a = ", a, "y = ", y)
```

What happens when we run this script?

```{code-cell}
:hide-output: false

%run test.py
```

First,

- The global namespace `{}` is created.  
- The function object is created, and `g` is bound to it within the global namespace.  
- The name `a` is bound to `0`, again in the global namespace.  
  
Next `g` is called via `y = g(10)`, leading to the following sequence of actions

- The local namespace for the function is created.  
- Local names `x` and `a` are bound, so that the local namespace becomes `{'x': 10, 'a': 1}`.  


Note that the global `a` was not affected by the local `a`.
  
- Statement `x = x + a` uses the local `a` and local `x` to compute `x + a`, and binds local name `x` to the result.  
- This value is returned, and `y` is bound to it in the global namespace.  
- Local `x` and `a` are discarded (and the local namespace is deallocated).  

  


+++

### Mutable Versus Immutable Parameters

This is a good time to say a little more about mutable vs immutable objects.

Consider the code segment

```{code-cell}
:hide-output: false

def f(x):
    x = x + 1
    return x

x = 1
print(f(x), x)
```

We now understand what will happen here: The code prints `2` as the value of `f(x)` and `1` as the value of `x`.

First `f` and `x` are registered in the global namespace.

The call `f(x)` creates a local namespace and adds `x` to it, bound to `1`.

Next, this local `x` is rebound to the new integer object `2`, and this value is returned.

None of this affects the global `x`.

However, it’s a different story when we use a **mutable** data type such as a list

```{code-cell}
:hide-output: false

def f(x):
    x[0] = x[0] + 1
    return x

x = [1]
print(f(x), x)
```

This prints `[2]` as the value of `f(x)` and *same* for `x`.

Here’s what happens

- `f` is registered as a function in the global namespace  
- `x` is bound to `[1]` in the global namespace  
- The call `f(x)`  
  - Creates a local namespace  
  - Adds `x` to the local namespace, bound to `[1]`  

We can see the identity of local `x` and the identity of global `x` are the same

```{code-cell}
:hide-output: false

def f(x):
    x[0] = x[0] + 1
    print(f'the identity of local x is {id(x)}')
    return x

x = [1]
print(f'the identity of global x is {id(x)}')
print(f(x), x)
```

- Within `f(x)`  
  - The list `[1]` is modified to `[2]`  
  - Returns the list `[2]`  


- The local namespace is deallocated, and the local `x` is lost  

  
If you want to modify the local `x` and the global `x` separately, you can create a [*copy*](https://docs.python.org/3/library/copy.html) of the list and assign the copy to the local `x`.

We will leave this for you to explore.


