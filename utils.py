"""
utils.py

This file contains the data structures Vector and Matrix and their supporting functions

@author: Anushree Das (ad1707)
"""

from collections import namedtuple

Vector = namedtuple('Vector', 'data')
Matrix = namedtuple('Matrix', ['rows', 'cols', 'data'])


def makeVector(n, f):
    """
    Returns a Vector named tuple that contains n copies of that float value
    :param n: number of elements
    :param f: function which returns a float value
    :return:  Vector
    """
    arr = []
    for i in range(n):
        arr.append(f())
    v = Vector(data=arr)
    return v


def setVec(v1,v2):
    """
    Mutates the first vector so that its contents is the same as the contents of the second
    :param v1: first vector
    :param v2: second vector
    :return:   None
    """
    v1.data.clear()
    for i in range(len(v2.data)):
        v1.data.append(v2.data[i])


def makeMatrix(n, m, f):
    """
    Returns a Matrix named tuple that is filled with that float value
    :param n: number of rows
    :param m: number of columns
    :param f: function which returns a float value
    :return:  Matrix
    """
    arr = []
    for i in range(n * m):
        arr.append(f())
    m = Matrix(rows=n, cols=m, data=arr)
    return m


def setMat(m1,m2):
    """
    Mutates the first matrix so that its contents is the same as the contents of the second
    :param m1: first matrix
    :param m2: second matrix
    :return:   None
    """
    m1.data.clear()
    for i in range(len(m2.data)):
        m1.data.append(m2.data[i])


def augmentColMat(m):
    """
    Takes a column matrix and returns its extension:
    the column matrix with a 1 at the bottom/end.
    :param m: matrix
    :return: augmented matrix
    """
    if m.cols != 1:
        raise TypeError('Not a column matrix!')
    arr = m.data.copy()
    arr.append(1)
    m = Matrix(rows=m.rows + 1, cols=m.cols, data=arr)
    return m


def colMatrixFromVector(v):
    """
    Takes a vector and returns the column matrix that represents the vector
    :param v: vector
    :return:  matrix that represents the vector
    """
    arr = v.data.copy()
    m = Matrix(rows=len(arr), cols=1, data=arr)
    return m


def vectorFromColMatrix(m):
    """
    Takes a column matrix and returns the vector that represents the column matrix
    :param m: matrix
    :return:  vector that represents the column matrix
    """
    if m.cols != 1:
        raise TypeError('Not a column matrix!')
    arr = m.data.copy()
    v = Vector(data=arr)
    return v


def mapMatrix(f, m):
    """
    Takes a function f and a matrix A and
    returns the matrix f(A)[i,j], where f(A)[i,j] = f(a[i,j])
    :param f: function
    :param m: matrix
    :return: mapped matrix
    """
    arr = m.data
    newArr = []
    for i in range(m.rows * m.cols):
        newArr.append(f(arr[i]))
    newM = Matrix(rows=m.rows, cols=m.cols, data=newArr)
    return newM


# override + operator for Vector
Vector.__add__ = lambda x, y: Vector([sum(val) for val in zip(x.data, y.data)])
# override + operator for Matrix
Matrix.__add__ = lambda x, y: Matrix(x.rows, x.cols, [sum(val) for val in zip(x.data, y.data)])


def add(x, y):
    """
    Takes two arguments of the same type,
    where this type can be either int, float, Vector, or Matrix.
    It returns a new object of the same type that is the sum of its arguments
    :param x: first variable
    :param y: second variable
    :return: sum of x and y
    """
    if type(x) is not type(y):
        raise TypeError('incompatible types for addition')
    if isinstance(x, Vector):
        if len(x.data) != len(y.data):
            raise TypeError('incompatible Vector dimensions')
    if isinstance(x, Matrix):
        if x.rows != y.rows or x.cols != y.cols or len(x.data) != len(y.data):
            raise TypeError('incompatible Matrix dimensions')

    z = x + y
    return z


# override - operator for Vector
Vector.__sub__ = lambda x, y: Vector([xi - yi for xi, yi in zip(x.data, y.data)])
# override - operator for Matrix
Matrix.__sub__ = lambda x, y: Matrix(x.rows, x.cols, [xi - yi for xi, yi in zip(x.data, y.data)])


def subtract(x, y):
    """
    Takes two arguments of the same type,
    where this type can be either int, float, Vector, or Matrix.
    It returns a new object of the same type that is the difference of its arguments
    :param x: first variable
    :param y: second variable
    :return: difference of x and y
    """
    if type(x) is not type(y):
        raise TypeError('incompatible types for subtraction')
    if isinstance(x, Vector):
        if len(x.data) != len(y.data):
            raise TypeError('incompatible Vector dimensions')
    if isinstance(x, Matrix):
        if x.rows != y.rows or x.cols != y.cols or len(x.data) != len(y.data):
            raise TypeError('incompatible Matrix dimensions')

    z = x - y
    return z


# override * operator for multiplying int with Vector
Vector.__rmul__ = lambda x, n: Vector([n * xi for xi in x.data])
# override * operator for multiplying int with Matrix
Matrix.__rmul__ = lambda x, n: Matrix(x.rows, x.cols, [n * xi for xi in x.data])


def scale(x, y):
    """
    Returns a new object of the same type as the second argument
    that is the second argument scaled by first argument
    :param x: integer
    :param y: second variable
    :return: y scaled by x
    """
    if x.is_integer():
        raise TypeError('first parameter should be an integer')

    z = x * y
    return z


def mult(m1, m2):
    """
    Performs matrix multiplication AxB
    :param m1: first matrix
    :param m2: second matrix
    :return: matrix resulted by the matrix multiplication
    """
    if isinstance(m1, Matrix) and isinstance(m2, Matrix):
        if m1.cols != m2.rows:
            raise TypeError('incompatible Matrices for multiplication')

        arr = []
        for ai in range(m1.rows):
            for bj in range(m2.cols):
                sum = 0
                for (aj, bi) in zip(range(m1.cols), range(bj, m2.rows * m2.cols, m2.cols)):
                    sum += m1.data[(ai * m1.cols) + aj] * m2.data[bi]
                arr.append(sum)

        m = Matrix(m1.rows, m2.cols, arr)

        return m


# override * operator for multiplying Vector with Vector
Vector.__mul__ = lambda x, y: Vector([xi * yi for xi, yi in zip(x.data, y.data)])
# override * operator for multiplying Matrix with Matrix
Matrix.__mul__ = lambda x, y: Matrix(x.rows, x.cols, [xi * yi for xi, yi in zip(x.data, y.data)])


def pointProd(x, y):
    """
    Returns the pointwise, or Hadamard, product
    :param x: first variable
    :param y: second variable
    :return: pointwise product of x and y
    """
    if type(x) is not type(y):
        raise TypeError('incompatible types for Hadamard product')
    if isinstance(x, Vector):
        if len(x.data) != len(y.data):
            raise TypeError('incompatible Vector dimensions')
    if isinstance(x, Matrix):
        if x.rows != y.rows or x.cols != y.cols or len(x.data) != len(y.data):
            raise TypeError('incompatible Matrix dimensions')

    z = x * y
    return z


def transpose(m):
    """
    Returns transpose of a matrix
    :param m: matrix
    :return: transpose of the matrix
    """
    arr = []
    for i in range(m.cols):
        for j in range(m.rows):
            arr.append(m.data[(j * m.cols) + i])

    transposeM = Matrix(m.cols, m.rows, arr)

    return transposeM


def dot(x, y):
    """
    Takes two arguments of the same type,
    where this type can be either int, float, Vector, or Matrix.
    It returns the dot product of the arguments.
    :param x: first variable
    :param y: second variable
    :return: dot product of x and y
    """
    if type(x) is not type(y):
        raise TypeError('incompatible types for  dot product')

    if isinstance(x, Vector):
        if len(x.data) != len(y.data):
            raise TypeError('incompatible Vector dimensions')
        else:
            x = colMatrixFromVector(x)
            y = colMatrixFromVector(y)

    if isinstance(x, Matrix):
        if x.rows != y.rows:
            raise TypeError('incompatible Matrix dimensions')
        else:
            x = transpose(x)
        z = mult(x, y).data

    else:
        z = x * y

    return z


def outerProd(x, y):
    """
    takes two arguments of the same type,
    where this type can be either int, float, Vector, or Matrix.
    It returns the outer product of the arguments.
    :param x: first variable
    :param y: second variable
    :return: outer product of x and y
    """
    # if type(x) is not type(y):
    #     raise TypeError('incompatible types for  dot product')

    if isinstance(x, Vector):
        x = colMatrixFromVector(x)

    if isinstance(y, Vector):
        y = colMatrixFromVector(y)

    if isinstance(x, Matrix):
        # print(x,y)
        if x.cols != y.cols:
            raise TypeError('incompatible Matrix dimensions')
        else:
            y = transpose(y)
        z = mult(x, y)

    else:
        z = Matrix(1, 1, x * y)

    return z