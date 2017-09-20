import matplotlib.pyplot as plt


def main(path):
    with open(path) as f:
        x, y = [], []
        parameter = ""
        for line in f.read().splitlines():
            if "," in line:  # "mae" in line -> parse parameter name
                if "mae" in line:  # values
                    if parameter:  # not empty
                        plot(x, y, parameter)
                        x, y = [], []
                    else: # parameter is set
                        parameter = line.split(',')[0]

                else:  # "mae" not in line
                    parts = line.split(',')
                    x.append(parts[0])
                    y.append(parts[1])

        plot(x, y, parameter)


def plot(x, y, parameter):
    min_y = min(y)
    min_x = x[y.index(min_y)]

    plt.plot(x, y)
    plt.scatter(min_x, min_y)
    plt.title("Different values for parameter {}".format(parameter))
    plt.show()


if __name__ == '__main__':
    main('output_content_only.txt')
