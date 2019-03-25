import numpy
v = numpy.sin(numpy.arange(20).reshape(4,5))
print(v)
v.argmax(axis=0)