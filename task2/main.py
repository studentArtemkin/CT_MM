
class Identer:
    def print(self,input):
        print(input)


with Indenter() as indent:
    indent.print('hi!')
    with indent:
        indent.print('hello')
    #        with indent:
    #            indent.print('bonjour')
    #indent.print('hey')