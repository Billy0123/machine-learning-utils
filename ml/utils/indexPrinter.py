class indexPrinter:
    def __init__(self):
        self.index = 0

    def print(self, input=None):
        self.index += 1
        print('%d:\n%s\n-----' % (self.index, input))