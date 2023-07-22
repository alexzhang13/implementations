#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main()
{
    int n;
    cin >> n;

    if (((1ll) * n * (n + 1)) % 4 != 0)
    {
        cout << "NO" << endl;
        return 0;
    }
    else
    {
        cout << "YES" << endl;
    }
    ll target = (1ll) * n * (n + 1) / 4;
    int stop_idx = -1;
    int wild = -1;

    for (int k = n; k > 0; k--)
    {
        if (target == 0)
            break;
        if (target - k < 0)
        {
            wild = target;
            break;
        }
        target -= k;
        stop_idx = k;
    }

    // first group
    int amt = n - stop_idx + 1;
    if (wild > 0)
        amt++;
    cout << amt << endl;
    if (wild > 0)
        cout << wild << " ";

    for (int k = n; k >= stop_idx; k--)
    {
        cout << k << " ";
    }
    cout << endl;

    // second group
    cout << n - amt << endl;
    for (int i = 1; i < stop_idx; i++)
    {
        if (i != wild)
            cout << i << " ";
    }
    cout << endl;

    return 0;
}