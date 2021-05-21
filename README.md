# Back Propagation Network
  Implements a back-propagation network

Usage:
## Create Back Propagation Network:
bp = makeBackProp(eta, N, fn, trace)
where eta is learning rate, N is list of the number of intended inputs for each layer of the network, fn is initialization thunk and trace is trace value.

## Train Back Propagation Network:
train(bp,dataset,epochs)

## Apply on single sample input:
applyBackPropVec(bp,inputVector)