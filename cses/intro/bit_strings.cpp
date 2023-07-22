#include <bits/stdc++.h>

#define ll long long

using namespace std;

const int MOD = 1e9 + 7;

ll binpow(int n, int mod)
{
    if (n == 0)
        return 1;
    if (n == 1)
        return 2;
    ll res = binpow(n / 2, mod);
    if (n & 1)
    {
        return ((res % mod) * (res % mod) * 2) % mod;
    }
    else
    {
        return ((res % mod) * (res % mod)) % mod;
    }
}

int main()
{
    int n;
    cin >> n;

    cout << binpow(n, MOD) << endl;
    return 0;
}