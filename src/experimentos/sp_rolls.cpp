#include <bits/stdc++.h>
#include "bucket_queue.hpp"

using namespace std;

/*
g++ -O3 \
    -march=native \
    -DNDEBUG \
    -std=c++20 \
    sp_rolls.cpp \
    -o sp_rolls
*/

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
    double min_edge;  // δ
    long long m;
};

// =====================================================
// std::priority_queue (double)                         
// =====================================================

double run_all_pairs_std(const Graph& g) {
    int n = (int)g.size();
    vector<double> dist(n);

    auto t0 = chrono::high_resolution_clock::now();

    for (int s = 0; s < n; s++) {
        fill(dist.begin(), dist.end(), numeric_limits<double>::infinity());

        priority_queue<
            pair<double,int>,
            vector<pair<double,int>>,
            greater<pair<double,int>>
        > pq;

        dist[s] = 0.0;
        pq.push({0.0, s});

        while (!pq.empty()) {
            auto [du, u] = pq.top();
            pq.pop();

            if (du > dist[u] + 1e-12) continue;

            for (const auto& e : g[u]) {
                double nd = du + e.w;
                if (nd < dist[e.to]) {
                    dist[e.to] = nd;
                    pq.push({nd, e.to});
                }
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1 - t0).count();
}

// =====================================================
// bucket queue (double nativo, método do artigo)
// =====================================================

double run_all_pairs_bucket(const Graph& g, double delta, double max_w) {
    int n = (int)g.size();
    vector<double> dist(n, 1e18);

    // C/δ é a chave inteira máxima possível — dimensiona a fila corretamente
    long long C = (long long)(max_w / delta);

    auto t0 = chrono::high_resolution_clock::now();

    constexpr int LEVELS = 3;
    bucket_queue<pair<double, unsigned int>> pq(C, n, LEVELS, delta);

    for (int s = 0; s < n; s++) {
        fill(dist.begin(), dist.end(), 1e18);
        pq.reset();

        dist[s] = 0.0;
        pq.push({0.0, (unsigned)s});

        while (!pq.empty()) {
            auto [du, u] = pq.top();
            pq.pop();

            // stale check com tolerância float
            if (du > dist[u] + 1e-12) continue;

            for (const auto& e : g[u]) {
                double nd = du + e.w;
                if (nd < dist[e.to] - 1e-12) {
                    dist[e.to] = nd;
                    pq.decrease_key(e.to, nd);
                }
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1 - t0).count();
}

// =====================================================
// leitura do grafo (só double, sem SCALE)
// =====================================================

LoadedGraphs load_graph(const string& filename) {
    ifstream fin(filename);
    if (!fin) throw runtime_error("Nao foi possivel abrir arquivo");

    int n;
    long long m;
    fin >> n >> m;

    LoadedGraphs out;
    out.g.resize(n);
    out.m = m;
    out.max_edge = 0.0;
    out.min_edge = numeric_limits<double>::infinity();

    for (long long i = 0; i < m; i++) {
        int u, v;
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
// main
// =====================================================

int main() {
    vector<int> sizes = {1000,2000,5000,10000};

    cout << "n,m,std_time,bucket_time,speedup,delta,C_ratio\n";
    cout << fixed << setprecision(6);

    for (int n_expected : sizes) {
        string filename = "graphs/graph_" + to_string(n_expected) + ".txt";

        try {
            auto data = load_graph(filename);
            int n = (int)data.g.size();

            cerr << "Running N=" << n
                 << " delta=" << data.min_edge
                 << " C/delta=" << (long long)(data.max_edge / data.min_edge)
                 << "...\n";

            double t_std    = run_all_pairs_std(data.g);
            double t_bucket = run_all_pairs_bucket(data.g, data.min_edge, data.max_edge);

            cout << n << ","
                 << data.m << ","
                 << t_std << ","
                 << t_bucket << ","
                 << (t_std / t_bucket) << ","
                 << data.min_edge << ","
                 << (long long)(data.max_edge / data.min_edge)
                 << "\n";
        }
        catch (const exception& e) {
            cerr << "Erro ao abrir " << filename << ": " << e.what() << "\n";
        }
    }

    return 0;
}