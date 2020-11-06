class Cat:
    def __init__(self, name):
        self.name = name

    def greet(self, fellow):
        print("Hello I am {}! I see you are also a cool fluffy kitty {}, let’s together purr at the human "
              "so they shall give us food.".format(self.name, fellow))
