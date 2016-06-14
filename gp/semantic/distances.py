import numpy


def root_mean_square_error(a, b, axis=0):
    squared_errors = numpy.square(a - b)
    mse = numpy.mean(squared_errors, axis=axis)
    if not numpy.isfinite(mse):
        return numpy.inf
    return numpy.sqrt(mse).item()


def cumulative_absolute_difference(a, b, axis=0):
    return numpy.sum(numpy.abs(a - b), axis=axis)
