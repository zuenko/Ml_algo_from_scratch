import numpy as np


def mse(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return sum((y_true - y_pred) ** 2) / len(y_true)


def Loss(a, b, points):
    y_true = [points[i][0] for i in range(len(points))]
    y_pred = [(a * points[i][1] + b) for i in range(len(points))]
    return (mse(y_true, y_pred))


def gradient_step(a_current, b_current, points, learning_rate):
    a_gradient = 0
    b_gradient = 0
    N = len(points)

    for i in range(N):
        y = points[i][0]
        x = points[i][1]
        a_gradient += float(2 / N) * (-x * (y - (a_current * x + b_current)))
        b_gradient += float(2 / N) * (-(y - (a_current * x + b_current)))

    new_a = a_current - (learning_rate * a_gradient)
    new_b = b_current - (learning_rate * b_gradient)
    return [new_a, new_b]


def gradient(points, starting_a, starting_b, learning_rate, num_iterations):
    a, b = starting_a, starting_b
    loss = 0
    k = 0
    for i in range(num_iterations):
        loss_itr = Loss(a, b, points)
        if loss < loss_itr or (loss_itr != float('Inf') and k < 10):
            loss = loss_itr
            a_step, b_step = a, b
            k = 0
        else:
            k += 1

            if loss_itr != float('Inf'):
                break

        a, b = gradient_step(a, b, np.array(points), learning_rate)
        print("Itr_num: {0}, Loss: {1}".format(i, round(loss, 4)))

    print("Last loss: {0}".format(round(loss, 5)))
    return round(a_step, 5), round(b_step, 5)


if __name__ == '__main__':
    with open('data.txt', 'r') as f:
        data = []
        for line in f.readlines():
            y, x = line.split(',')
            data.append([float(y), float(x)])

    num_iterations = 10000000
    print("After {0} iterations a, b = {1}".format(num_iterations, gradient(data, 0, 0, 1e-4, num_iterations)))
