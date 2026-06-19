#include <bits/stdc++.h>
#include "bucket_queue.hpp"

using namespace std;

constexpr long long SCALE = 1000LL;

/*
g++ -O3 \
    -march=native \
    -DNDEBUG \
    -std=c++20 \
    sp_fotos.cpp \
    -o sp_fotos
*/

// =====================================================
// Grafos
// =====================================================

struct EdgeDouble {
    int to;
    double w;
};

struct EdgeInt {
    int to;
    long long w;
};

using GraphDouble = vector<vector<EdgeDouble>>;
using GraphInt = vector<vector<EdgeInt>>;


// =====================================================
// std::priority_queue
// =====================================================

double run_all_pairs_std(const GraphDouble& g) {

    int n = (int)g.size();
    vector<double> dist(n);

    auto t0 = chrono::high_resolution_clock::now();

    for (int s = 0; s < n; s++) {

        fill(dist.begin(), dist.end(),
             numeric_limits<double>::infinity());

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

            if (du != dist[u]) continue;

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
// bucket queue
// =====================================================

double run_all_pairs_bucket(const GraphInt& g, long long C) {

    int n = (int)g.size();
    vector<long long> dist(n);

    auto t0 = chrono::high_resolution_clock::now();

    constexpr int LEVELS = 3;

    bucket_queue<pair<long long,unsigned int>> pq(C, n, LEVELS);

    for (int s = 0; s < n; s++) {

        fill(dist.begin(), dist.end(), LLONG_MAX);

        pq.reset();

        dist[s] = 0;
        pq.push({0, (unsigned)s});

        while (!pq.empty()) {

            auto [du, u] = pq.top();
            pq.pop();

            if (du != dist[u]) continue;

            for (const auto& e : g[u]) {

                long long nd = du + e.w;

                if (nd < dist[e.to]) {

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
// leitura do grafo
// =====================================================

struct LoadedGraphs {
    GraphDouble g_double;
    GraphInt g_int;

    long long max_edge = 0;
    long long m = 0;
    int n = 0;
};

LoadedGraphs load_graph(const string& filename) {

    ifstream fin(filename);

    if (!fin) {
        throw runtime_error("Nao foi possivel abrir arquivo: " + filename);
    }

    int n;
    long long m;

    fin >> n >> m;

    LoadedGraphs out;
    out.n = n;
    out.m = m;

    out.g_double.resize(n);
    out.g_int.resize(n);

    for (long long i = 0; i < m; i++) {

        int u, v;
        double w;

        fin >> u >> v >> w;

        long long wi = llround(w * SCALE);

        out.max_edge = max(out.max_edge, wi);

        out.g_double[u].push_back({v, w});
        out.g_double[v].push_back({u, w});

        out.g_int[u].push_back({v, wi});
        out.g_int[v].push_back({u, wi});
    }

    return out;
}


// =====================================================
// lista automática de arquivos
// =====================================================

vector<string> list_graph_files(const string& folder) {

    vector<string> files;

    for (const auto& entry :
         filesystem::directory_iterator(folder)) {

        if (entry.is_regular_file()) {
            string path = entry.path().string();

            if (path.find(".txt") != string::npos) {
                files.push_back(path);
            }
        }
    }

    sort(files.begin(), files.end());
    return files;
}


// =====================================================
// main
// =====================================================

int main() {

    string folder = "graphs";

    auto files = list_graph_files(folder);

    cout << "n,m,std_time,bucket_time,speedup\n";
    cout << fixed << setprecision(6);

    for (const auto& filename : files) {

        try {

            auto data = load_graph(filename);

            long long C = data.max_edge;

            cerr << "Running " << filename
                 << " (n=" << data.n << ")\n";

            double t_std =
                run_all_pairs_std(data.g_double);

            double t_bucket =
                run_all_pairs_bucket(data.g_int, C);

            cout << data.n << ","
                 << data.m << ","
                 << t_std << ","
                 << t_bucket << ","
                 << (t_std / t_bucket)
                 << "\n";

        } catch (const exception& e) {

            cerr << "Erro em " << filename
                 << ": " << e.what() << "\n";
        }
    }

    return 0;
}