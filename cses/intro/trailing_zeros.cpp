#include <bits/stdc++.h>

using namespace std;
#define ll long long

int main()
{
    ios::sync_with_stdio(true);
    cin.tie(0);
    int n;

    cin >> n;

    int total = 0;
    while (n > 0)
    {
        n = n / 5;
        total += n;
    }
    cout << total << endl;

    return 0;
}
