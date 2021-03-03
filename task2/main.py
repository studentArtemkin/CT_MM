class Indenter(object):
    def __init__(self):
        self.state = -1
    def __enter__(self):
        self.state += 1
        return self
    def __exit__(self, type, value, traceback):
        self.state -= 1
    def print0(self, input_string):
        print("\t" * self.state + input_string)


with Indenter() as indent:
    indent.print0('hi!')
    with indent:
        indent.print0('hello')
        with indent:
            indent.print0('bonjour')
    indent.print0('hey')