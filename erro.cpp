#include <bits/stdc++.h>
#include "bucket_queue.hpp"

/*
g++ -O3 \
    -march=native \
    -DNDEBUG \
    -std=c++20 \
    erro.cpp \
    -o erro
*/

using namespace std;

// =====================================================
// Grafos
// =====================================================

struct Edge {
    int to;
    double w;
};

using Graph = vector<vector<Edge>>;

struct LoadedGraphs {
    Graph g;
    double max_edge;
    double min_edge;
    long long m;
};

// =====================================================
// Dijkstra std::priority_queue
// =====================================================

vector<double> sssp_std(const Graph& g, int s)
{
    int n = (int)g.size();

    vector<double> dist(n, numeric_limits<double>::infinity());

    priority_queue<
        pair<double,int>,
        vector<pair<double,int>>,
        greater<pair<double,int>>
    > pq;

    dist[s] = 0.0;
    pq.push({0.0, s});

    while(!pq.empty()){

        auto [du,u] = pq.top();
        pq.pop();

        if(du > dist[u] + 1e-12) continue;

        for(const auto& e : g[u]){

            double nd = du + e.w;

            if(nd < dist[e.to] - 1e-12){
                dist[e.to] = nd;
                pq.push({nd, e.to});
            }
        }
    }

    return dist;
}

// =====================================================
// bucket queue (double)
// =====================================================

vector<double> sssp_bucket(const Graph& g, int s, double delta, double max_w)
{
    int n = (int)g.size();

    vector<double> dist(n, 1e18);

    long long C = (long long)(max_w / delta);

    constexpr int LEVELS = 3;

    bucket_queue<pair<double,unsigned int>> pq(C, n, LEVELS, delta);

    pq.reset();

    dist[s] = 0.0;
    pq.push({0.0, (unsigned)s});

    while(!pq.empty()){

        auto [du,u] = pq.top();
        pq.pop();

        if(du > dist[u] + 1e-12) continue;

        for(const auto& e : g[u]){

            double nd = du + e.w;

            if(nd < dist[e.to] - 1e-12){
                dist[e.to] = nd;
                pq.decrease_key(e.to, nd);
            }
        }
    }

    return dist;
}

// =====================================================
// leitura do grafo
// =====================================================

LoadedGraphs load_graph(const string& filename)
{
    ifstream fin(filename);

    if(!fin){
        throw runtime_error("Nao foi possivel abrir arquivo");
    }

    int n;
    long long m;
    fin >> n >> m;

    LoadedGraphs out;
    out.g.resize(n);
    out.m = m;

    out.max_edge = 0.0;
    out.min_edge = numeric_limits<double>::infinity();

    for(long long i = 0; i < m; i++){

        int u,v;
        double w;
        fin >> u >> v >> w;

        out.max_edge = max(out.max_edge, w);
        out.min_edge = min(out.min_edge, w);

        out.g[u].push_back({v, w});
        out.g[v].push_back({u, w});
    }

    return out;
}

// =====================================================
// main (comparação completa)
// =====================================================

int main()
{
    vector<int> sizes = {5000};

    cout << "n,m,std_time,bucket_time,speedup,mismatches,total\n";
    cout << fixed << setprecision(6);

    for(int n_expected : sizes){

        string filename = "graphs/graph_" + to_string(n_expected) + ".txt";

        try {
            auto data = load_graph(filename);
            int n = (int)data.g.size();

            double delta = data.min_edge;
            double max_w = data.max_edge;

            cerr << "Running N=" << n
                 << " delta=" << delta
                 << " C/delta=" << (long long)(max_w / delta)
                 << "...\n";

            // -----------------------------
            // std all pairs
            // -----------------------------
            auto t0 = chrono::high_resolution_clock::now();

            vector<vector<double>> all_std(n);

            for(int s = 0; s < n; s++){
                all_std[s] = sssp_std(data.g, s);
            }

            auto t1 = chrono::high_resolution_clock::now();

            double t_std = chrono::duration<double>(t1 - t0).count();

            // -----------------------------
            // bucket all pairs + compare
            // -----------------------------
            auto t2 = chrono::high_resolution_clock::now();

            long long mismatches = 0;
            long long total_vals = 0;

            for(int s = 0; s < n; s++){

                auto d_bucket = sssp_bucket(data.g, s, delta, max_w);

                for(int v = 0; v < n; v++){

                    if(all_std[s][v] == numeric_limits<double>::infinity())
                        continue;

                    if(fabs(all_std[s][v] - d_bucket[v]) > 1e-6){
                        mismatches++;
                    }

                    total_vals++;
                }
            }

            auto t3 = chrono::high_resolution_clock::now();

            double t_bucket = chrono::duration<double>(t3 - t2).count();

            cout << n << ","
                 << data.m << ","
                 << t_std << ","
                 << t_bucket << ","
                 << (t_std / t_bucket) << ","
                 << mismatches << ","
                 << total_vals << "\n";

        }
        catch(const exception& e){
            cerr << "Erro ao abrir " << filename << ": " << e.what() << "\n";
        }
    }

    return 0;
}