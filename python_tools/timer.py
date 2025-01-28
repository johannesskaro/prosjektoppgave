from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def main():
    with timer() as time: 
        print("Here we do some stuff that takes time. ")
    print(f"The stuff took: {time()}")

if __name__ == "__main__":
    main()
