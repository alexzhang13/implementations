#include <bits/stdc++.h>

using namespace std;

class Bitmask
{
private:
    int mask;

public:
    Bitmask()
    {
        this->mask = 0;
    }
    int get(int idx)
    {
        return (this->mask & (1 << idx));
    }
    void set(int idx)
    {
        this->mask = this->mask | (1 << idx);
    }
    void unset(int idx)
    {
        this->mask = this->mask & ~(1 << idx);
    }
    void flip()
    {
        this->mask = ~this->mask;
    }
    const int operator[](int idx)
    {
        return get(idx);
    }
};

int main()
{

    int n;
    std::cin >> n;

    Bitmask mask = Bitmask();
    mask.set(n);
    mask.set(5);

    for (int i = 0; i < 32; i++)
    {
        cout << mask[i] << " ";
    }
    cout << endl;

    return 0;
}