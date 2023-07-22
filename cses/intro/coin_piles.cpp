#include <bits/stdc++.h>

using namespace std;
#define ll long long

void solve()
{
    int a, b;
    cin >> a >> b;

    if (a > b)
        swap(a, b);
    if (b - a > 1)
    {
        int tmp = b;
        b -= (b - a - 1) * 2;
        a -= (tmp - a - 1);
        if (a <= 0)
        {
            cout << "NO" << endl;
            return;
        }
    }
    a = a % 3;
    b = b % 3;
    if ((a == 1 && b == 2) || (a == 0 && b == 0))
    {
        cout << "YES" << endl;
    }
    else
    {
        cout << "NO" << endl;
    }
}

int main()
{
    ios::sync_with_stdio(true);
    cin.tie(0);
    int t;

    cin >> t;
    while (t--)
    {
        solve();
    }

    return 0;
}