#include <bits/stdc++.h>
using namespace std;

typedef long long ll ;

struct bkt{
    int tail, sz;
};

struct element{
    pair<double,int> data;
    int prev, prox;
};

struct pool_list{

    element *pool;
    int *free_list;
    int *idxs;
    int free_top;

    pool_list() : free_list(nullptr), idxs(nullptr), free_top(0) {}
    pool_list(int c) {
        pool = new element[c];
        free_list = new int[c];
        idxs = new int[c];
        for(int i=0;i<c;i++) {
            free_list[i] = i;
            idxs[i] = -1;
        }
        free_top = c-1;
    }

    int alocar(pair<double,int> novo){
        int idx = free_list[free_top--];
        pool[idx].data = novo;
        pool[idx].prev = pool[idx].prox = -1;;
        return idx;
    }
    
    void insert(pair<double,int> novo, bkt *cauda){ 
        int idx =  alocar(novo);
        idxs[novo.second] = idx;
        int t = cauda->tail;
        if (t!=-1) {
            pool[t].prox = idx;
            pool[idx].prev = t;
        }
        cauda->tail = idx;
        cauda->sz++;
    }

    void pop(bkt *bucket){
        int t = bucket->tail;
        int ant = pool[t].prev;

        if (ant != -1) {
            pool[ant].prox = -1;
        }

        free_top++;
        free_list[free_top] = t;
        idxs[pool[t].data.second] = -1;

        bucket->tail = ant;
        bucket->sz--;
    }

    void remove(int u, bkt *bucket){
        int idx = idxs[u];
        int ant = pool[idx].prev;
        int next = pool[idx].prox;

        if(bucket->tail == idx) bucket->tail = pool[idx].prev;
        bucket->sz--;

        if (ant != -1) pool[ant].prox = next;
        if (next != -1) pool[next].prev = ant;

        pool[idx].prev = pool[idx].prox = -1;
        free_top++;
        free_list[free_top] = idx;
        idxs[u] = -1;
    }

    void clear(int n) {
        for(int i = 0; i < n; i++) { 
            free_list[i] = i; idxs[i] = -1;
        }
        free_top = n - 1;
    }

    void del(){
        delete[] pool;
        delete[] free_list;
        delete[] idxs;
    }
};