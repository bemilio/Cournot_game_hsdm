
import threading
import numpy as np
import time
class Agent:
    def __init__(self, id, neighbour):
        self.id = id
        self.neighbour = neighbour

    def threaded_fun(self,x):
        for t in range(10):
            c=x[self.id] + 1
            x[self.id]=c
            print("Hello, my ID is ", self.id, " and my current state is ", x[self.id])
            time.sleep(0.1)

    def run(self, x):
        self.thread = threading.Thread(target = self.threaded_fun, args=(x,))
        self.thread.start()

if __name__ == "__main__":
    a_1=Agent(0,1)
    a_2=Agent(1,0)
    x = np.array([0,10])
    a_1.run(x)
    a_2.run(x)
    print("All threads launched")
    time.sleep(2)
    print("Final x = ", x)