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
        return this->mask & (1 << idx);
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