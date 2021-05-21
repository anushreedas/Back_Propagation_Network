from utils import *
from math import exp
from random import random

BackProp = namedtuple('BackProp',['eta', 'weightColMats', 'trace'])


def makeBackProp(eta,N,fn,trace):
    """
    Returns a BackProp named tuple with the given parameters
    :param eta: learning rate
    :param N:   list of the number of intended inputs for each layer of the network
    :param fn:  initialization thunk
    :param trace: trace value
    :return:    BackProp named tuple
    """
    weightColMats = []
    # initialize weights for all layers along with bias
    for i in range(1,len(N)-1):
        weightColMats.append(makeMatrix(N[i] + 1, N[i-1] + 1, fn))
    # initialize weights for last layer without bias
    weightColMats.append(makeMatrix(N[-1], N[- 2] + 1, fn))

    return BackProp(eta=eta, weightColMats=weightColMats, trace=trace)


def sigma(x):
    """
    Implements the step function for backpropagation model.
    :param x:   value
    :return:
    """
    return 1/(1+(exp(-x)))


def applyBackPropAll(backProp,augColMatrix):
    """
    Returns the output of the backpropagation for each layer as a list of column matrices
    The first element of the list is the augmented input column matrix.
    The next column matrix is computed by multiplying the corresponding weight matrix
    by the column matrix, and then squashing the result.
    :param backProp:  backpropagation model
    :param augColMatrix:list of input vectors
    :return: list of column matrices
    """
    for element in augColMatrix.data:
        if element != 0 and element != 1:
            raise TypeError('The augmented column matrix should contain only 1s and 0s')

    result = []
    # append input vector
    result.append(augColMatrix)

    # result of each layer
    for i in range(len(backProp.weightColMats)):
        xl = dot(transpose(backProp.weightColMats[i]),result[-1])
        yl = []
        for ele in xl:
            # apply squashing function
            yl.append(sigma(ele))
        result.append(Matrix(backProp.weightColMats[i].rows, result[-1].cols,yl))

    return result


def applyBackPropVec(backProp,inputVector):
    """
    Returns the output of the backpropagation for last layer only
    :param backProp:  backpropagation model
    :param inputVector:input vector
    :return:
    """
    for element in inputVector.data:
        if element != 0 and element != 1:
            raise TypeError('The input vector should contain only 1s and 0s')

    # convert input vector to column matrix and augment the column matrix
    augColMatrix = augmentColMat(colMatrixFromVector(inputVector))

    # return result of last layer
    return vectorFromColMatrix(applyBackPropAll(backProp,augColMatrix)[-1])


def trainOnce(backProp,inputVector,targetOutput):
    """
    Applies the backpropagation learning rule to the backpropagation model
    :param backProp:  backpropagation model
    :param inputVector: sample input vector
    :param targetOutput:target output
    :return: None
    """
    if backProp.trace == 2:
        print('On sample: input=',inputVector,'target=',targetOutput,',',backProp)

    # convert input vector to column matrix and augment the column matrix
    augColMatrix = augmentColMat(colMatrixFromVector(inputVector))

    next_d_y = None
    for i in range(len(backProp.weightColMats)-1,-1,-1):
        # get result for all layers
        y = applyBackPropAll(backProp, augColMatrix)

        if i == len(backProp.weightColMats)-1:
            # if last layer
            d_y = subtract(vectorFromColMatrix(y[i+1]),targetOutput)
        else:
            y_i = vectorFromColMatrix(y[i+1])
            ones = makeVector(len(y_i.data), lambda: 1)
            temp = subtract(ones, y_i)
            y_x = dot(y_i, temp)[0]
            d_x = scale(y_x, next_d_y)
            d_y = colMatrixFromVector(Vector(dot(backProp.weightColMats[i+1],colMatrixFromVector(d_x))))

        next_d_y = d_y
        y_i = vectorFromColMatrix(y[i+1])
        ones = makeVector(len(y_i.data),lambda : 1)
        temp = subtract(ones,y_i)
        y_x = dot(y_i,temp)[0]

        x_W = vectorFromColMatrix(y[i])

        delta = - backProp.eta * outerProd(scale(y_x, d_y),x_W)

        # update weights for layer i
        setMat(backProp.weightColMats[i], add(backProp.weightColMats[i], delta))


def trainEpoch(backProp,dataset):
    """
    Trains the backpropagation once for each entry in the data set
    :param backProp:  backpropagation model
    :param dataset:     data set on which to train on
    :return:           None
    """
    # train for each row in dataset
    for sample in dataset:
        trainOnce(backProp, sample[0], sample[1])

    if backProp.trace == 1:
        print('After epoch: ', backProp)



def train(backProp,dataset,epochs):
    """
    Trains backpropagation on given dataset iteratively and
    terminates when either there is no change or
    the number of epochs exceeds the bound.
    :param backProp:  backpropagation model
    :param dataset:     data set on which to train on
    :param epochs:      number of training epochs
    :return:            None
    """
    iteration = 0

    # terminates when number of epochs exceeds the bound
    while iteration < epochs:
        trainEpoch(backProp,dataset)
        iteration += 1


def andOrDataSetCreator():
    """
    Creates AND dataset for backpropagation model
    :return: AND dataset
    """
    data = []
    for i in range(1,-1,-1):
        for j in range(1,-1,-1):
            data.append([ Vector(data=[i,j]) ,Vector(data=[i & j,i | j]) ])
    return data


andOrDataSet = andOrDataSetCreator()

def xorDataSetCreator():
    """
    Creates AND dataset for backpropagation model
    :return: AND dataset
    """
    data = []
    for i in range(1,-1,-1):
        for j in range(1,-1,-1):
            y = 1
            if i == j:
                y = 0
            data.append((Vector(data=[i,j]),Vector(data=[y])))
    return data


xorDataSet = xorDataSetCreator()

if __name__ == "__main__":

    andOrB = makeBackProp(.2,[2,2,2],(lambda : random()-.5),0)

    train(andOrB,andOrDataSet,5000)

    print(andOrB)

    print(applyBackPropVec(andOrB, Vector([1, 1])))
    print(applyBackPropVec(andOrB, Vector([0, 1])))
    print(applyBackPropVec(andOrB, Vector([1, 0])))
    print(applyBackPropVec(andOrB, Vector([0, 0])))

    xorB = makeBackProp(.2, [2, 10, 1], (lambda: random()), 0)

    train(xorB, xorDataSet, 5000)

    print(xorB)

    print(applyBackPropVec(xorB, Vector([1, 1])))
    print(applyBackPropVec(xorB, Vector([0, 1])))
    print(applyBackPropVec(xorB, Vector([1, 0])))
    print(applyBackPropVec(xorB, Vector([0, 0])))