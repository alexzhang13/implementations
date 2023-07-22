#include <bits/stdc++.h>

using namespace std;

#define ll long long

const int N = 10000;

int main()
{
    int n;
    cin >> n;

    for (int k = 1; k <= n; k++)
    {
        // (k^2 choose 2) - 2 * 2 * (k - 1) * (k - 2)
        cout << (1ll) * (k * k) * (k * k - 1) / 2 - 4 * (k - 1) * (k - 2) << endl;
    }

    return 0;
}