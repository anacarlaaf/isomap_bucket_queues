cdef extern from "<queue>" namespace "std" nogil:
    cdef cppclass queue[T]:
        queue() except +
        queue(queue&) except +
        #queue(Container&)
        T& back()
        bint empty()
        T& front()
        void pop()
        void push(T&)
        size_t size()
        # C++11 methods
        void swap(queue&)

    cdef cppclass priority_queue[T]:
        priority_queue() except +
        priority_queue(priority_queue&) except +
        #priority_queue(Container&)
        bint empty()
        void pop()
        void push(T&)
        size_t size()
        T& top()
        # C++11 methods
        void swap(priority_queue&)

    
cdef extern from "../filas/bucket_queue.hpp" nogil:
    cdef cppclass bucket_queue[T]:
        bucket_queue(long long c, int n, int niveis, double delta) except +
        bint   empty()
        void   pop()
        void   push(T&)
        size_t size()
        T&     top()
        void   reset()
        void   decrease_key(int v, double new_dist)