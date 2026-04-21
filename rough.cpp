#include <iostream>
#include <vector>
using namespace std;

const int MOD = 1e9 + 7;

long long modpow(long long a, long long b) {
    long long res = 1;
    while (b) {
        if (b & 1)
            res = res * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return res;
}

int countWinningSetups(int N, int K, int D) {
    int total = 0;
    int half = K / 2;

    vector<long long> fact(N + 1), inv_fact(N + 1);
    fact[0] = 1;
    for (int i = 1; i <= N; ++i) {
        fact[i] = fact[i - 1] * i % MOD;
    }

    inv_fact[N] = modpow(fact[N], MOD - 2);
    for (int i = N - 1; i >= 0; --i) {
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD;
    }

    auto comb = [&](int n, int k) {
        if (k < 0 || k > n)
            return 0LL;
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD;
    };

    for (int white_start = 1; white_start + 2 * (half - 1) <= N; ++white_start) {
        int white_end = white_start + 2 * (half - 1);

        for (int black_start = white_end + 2; black_start + 2 * (half - 1) <= N; ++black_start) {
            int black_end = black_start + 2 * (half - 1);

            int left_segment = white_start - 1;
            int middle_segment = black_start - white_end - 1;
            int right_segment = N - black_end;

            auto grundy = [D](int s) { return s % (D + 1); };
            int g_left = grundy(left_segment);
            int g_middle = grundy(middle_segment);
            int g_right = grundy(right_segment);

            int nim_sum = g_left ^ g_middle ^ g_right;

            if (nim_sum != 0) {
                int white_slots = (white_end - white_start) / 2 + 1;
                int black_slots = (black_end - black_start) / 2 + 1;
                long long ways = comb(white_end - white_start + 1, half) * comb(black_end - black_start + 1, half) % MOD;
                total = (total + ways) % MOD;
            }
        }
    }

    return total;
}

int main() {
    int N, K, D;
    cin >> N >> K >> D;
    cout << countWinningSetups(N, K, D) << endl;
    return 0;
}
