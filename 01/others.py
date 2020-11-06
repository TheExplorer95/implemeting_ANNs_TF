import numpy as np
np.set_printoptions(precision=3)

# list comp
squared_nums = [x**2 for x in range(101) if x**2 % 2 == 0]
print(squared_nums)


# generator
def meowrators():
    counter = 1
    while True:
        yield (counter * 'Meow ')
        counter *= 2


meow = meowrators()
for _ in range(5):
    print(next(meow))

# numpy slicing
five_times_five = np.random.normal(size=(5, 5))
print(five_times_five)
five_times_five[five_times_five**2 > 0.1] = 42
print(five_times_five)
print(five_times_five[:,2])