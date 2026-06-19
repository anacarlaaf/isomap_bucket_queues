#include<bits/stdc++.h>
using namespace std;
#include "define.hpp"

struct _klv_bucket_queue_DK{
    double delta;
    //double *dist_real;// TEMPORÁRIO
    pool_list pool;
    bkt **bucket;
    pair<int,int> *qBucket;  // (level, bucket_index) — ambos int
    int n_nodes;
    int *actBucket; // bucket ativo em cada nível
    ll *width;      // largura dos buckets de cada nível
    ll *lowerB;     // lower bounds de cada nível
    ll *upperB;     // upper bounds de cada nível
    int *sizeLv;    // qtd de elementos por nível 
    int r;          // rounds
    int _size;       // total na estrutura
    int k;          // níveis
    ll d;           // qtd de buckets por nível
 
    // inicializa as estruturas
    _klv_bucket_queue_DK(ll c, int n, int niveis, double delta_val) {
        delta = delta_val;
        k = niveis;
        r=0;
        _size = 0;
        d = ceil(pow((double)(c+1), 1.0 / k)) + 1;
        //fill(dist_real, dist_real + n, 1e18);
 
        actBucket  = new int[k]();
        width      = new ll[k]();
        lowerB     = new ll[k]();
        upperB     = new ll[k]();
        qBucket = new pair<int,int>[n];
        sizeLv     = new int[k]();
        bucket  = new bkt*[k];
        pool    = pool_list(n);
        n_nodes = n;
 
        width[0] = 1;
        for (int i=1;i<k;i++) width[i] = width[i-1]*d;
        
        lowerB[k-1] = 0;
        upperB[k-1] = d * width[k-1] - 1;
        for(int i=k-2;i>=0;i--) updBounds(i);
 
        for (int i=0;i<k;i++){
            bucket[i] = new bkt[d]();
            for (int j = 0; j < d; j++) bucket[i][j].tail = -1;
        }
        for(int i = 0; i < n; i++) qBucket[i] = {-1, -1};
    }
 
    // destroi estruturas
    ~_klv_bucket_queue_DK(){
        for(int i=0;i<k;i++){
            delete[] bucket[i];
        }
        delete[] bucket;
        delete[] actBucket;        
        delete[] width;
        delete[] lowerB;
        delete[] upperB;
        delete[] sizeLv;
        delete[] qBucket;
        //delete[] dist_real;
        pool.del();
    };
 
    void updTopBounds(){
        lowerB[k-1] = r * d * width[k-1];
        upperB[k-1] = lowerB[k-1] + d * width[k-1] - 1;
    }
 
    void updBounds(int lv){
        lowerB[lv] = lowerB[lv+1] + actBucket[lv+1] * width[lv+1];
        upperB[lv] = lowerB[lv] + d * width[lv] - 1;
    }
 
    void push(int v, double dist_f){
        ll key = (ll)(dist_f / delta);  // floor(D(v)/δ)

        for(int i = 0; i < k; i++){
            if(lowerB[i] <= key && key <= upperB[i]){
                int j = (int)((key - lowerB[i]) / width[i]);
                pool.insert({dist_f, v}, &bucket[i][j]);
                qBucket[v] = {i, j};
                sizeLv[i]++;
                _size++;
                return;
            }
        }
        // wrap-around: key > upperB[k-1]
        ll j = (key - upperB[k-1] - 1) / width[k-1];
        // j deve ser >= 0 e < d
        pool.insert({dist_f, v}, &bucket[k-1][(int)j]);
        qBucket[v] = {k-1, (int)j};
        sizeLv[k-1]++;
        _size++;
    }

    void update() {
        // procura bucket não vazio no bottom level

        while(actBucket[0] < d && bucket[0][actBucket[0]].sz==0){
            actBucket[0]++;
        }
        if(actBucket[0] < d) return;

 
        // encontra o nível mais baixo não vazio
        int f = 1;
        while(f < k && sizeLv[f] == 0) f++;
 
        // se top-level e ele está vazio, próximo round
        if(f == k-1) {
            while(actBucket[f] < d && bucket[f][actBucket[f]].sz==0){
                actBucket[f]++;
            }
 
            if(actBucket[f] == d) {
                r++;
                updTopBounds();
                actBucket[f] = 0;
                while(actBucket[f] < d && bucket[f][actBucket[f]].sz==0){
                    actBucket[f]++;
                }
            }
            // atualiza bounds dos níveis abaixo antes de expandir
            for(int idx = k-2; idx >= 0; idx--) updBounds(idx);
        }
 
        // expande do nível f até o nível 1
        for(int i = f; i > 0; i--) {
            while(actBucket[i] < d && bucket[i][actBucket[i]].sz==0)
                actBucket[i]++;
  
            // atualiza os bounds do nível abaixo
            updBounds(i-1);
            int new_act = d;
 
            // distribui elementos no nível abaixo
            int src = actBucket[i];
            while(bucket[i][src].sz) {
                int idx = bucket[i][src].tail;
                pair<double,int> elem = pool.pool[idx].data;
                pool.pop(&bucket[i][src]);
                sizeLv[i]--;

                ll key = (ll)(elem.first / delta);
                int target = (int)((key - lowerB[i-1]) / width[i-1]); 
                
                new_act = min(new_act, target);
                pool.insert(elem, &bucket[i-1][target]);
                qBucket[elem.second] = {i-1, target};
                sizeLv[i-1]++;
            }
 
            if(new_act < d) actBucket[i-1] = new_act;
        }
    }

    pair<double,int> top(){
        update();

        return pool.pool[bucket[0][actBucket[0]].tail].data;
    }
 
    void pop(){
        //update();
        pool.pop(&bucket[0][actBucket[0]]);
        sizeLv[0]--;
        _size--;
    }
 
    bool empty() { return _size == 0; }
 
    void decrease_key(int u, double new_du){
        if(pool.idxs[u]!=-1){
            pair<int,int> loc = qBucket[u];
            pool.remove(u, &bucket[loc.first][loc.second]);
            sizeLv[loc.first]--;
            _size--;
        }
        push(u, new_du);
    }

    void reset() {
        _size = 0;
        r = 0;
        for (int i = 0; i < k; i++) {
            actBucket[i] = 0;
            sizeLv[i] = 0;
            for (int j = 0; j < d; j++) {
                bucket[i][j].tail = -1;
                bucket[i][j].sz = 0;
            }
        }
        lowerB[k-1] = 0;
        upperB[k-1] = d * width[k-1] - 1;
        for (int i = k-2; i >= 0; i--) updBounds(i);  // necessário após reset
        pool.clear(n_nodes);
        //for (int i = 0; i < n_nodes; i++) qBucket[i] = {-1, -1};  // evita leitura de lixo no decrease_key
    }

};

// Wrapper template para compatibilidade com Cython (queue.pxd declara bucket_queue[T])
// T = pair<long long, unsigned int> onde first=dist, second=vertex
template<typename T>
struct bucket_queue : public _klv_bucket_queue_DK {
    bucket_queue(long long c, int n, int niveis, double delta_val)
        : _klv_bucket_queue_DK(c, n, niveis, delta_val) {}

    void push(const T& val) {
        _klv_bucket_queue_DK::push((int)val.second, (double)val.first);
    }

    T top() {
        pair<double,int> p = _klv_bucket_queue_DK::top();
        return T(p.first, (unsigned int)p.second);
    }

    void pop() {
        _klv_bucket_queue_DK::pop();
    }

    size_t size() const {
        return (size_t)(this->_size);
    }

    bool empty() {
        return _klv_bucket_queue_DK::empty();
    }

    void decrease_key(int v, double new_dist){
        _klv_bucket_queue_DK::decrease_key(v, new_dist);
    }

    void reset(){
        _klv_bucket_queue_DK::reset();
    }
};