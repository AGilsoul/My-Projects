//
// Created by agils on 11/19/2022.
//

#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>

const int PRIME = 31;
const int M = 1e9 + 7;

using std::string;
using std::vector;
using std::ostream;
using std::endl;

class hashtable {
public:
    hashtable() {}

    hashtable(int numItems) {
        if (numItems < 3000) {
            size = 3001;
        }
        else {
            size = nextPrime(numItems);
        }
        hashValues = vector<int>(size, -1);
    }

    bool insert(string key, int value) {
        int hashCode = hashFunc(key);
        if (hashValues[hashCode] != -1) {
            return false;
        }
        hashValues[hashCode] = value;
        return true;
    }

    int& operator[](string key) {
        int hashCode = hashFunc(key);
        return hashValues[hashCode];
    }

    friend ostream& operator<<(ostream& out, const hashtable& h) {
        for (int i = 0; i < h.hashValues.size(); i++) {
            out << i << " : " << h.hashValues[i] << endl;
        }
        return out;
    }

private:

    int hashFunc(string key) {
        int hashCode = 0;
        for (int i = 0; i < key.length(); i++) {
            hashCode += int(key[i] * pow(PRIME, i)) % size;
        }
        hashCode %= size;
        return hashCode;
    }

    bool isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;

        if (n % 2 == 0 || n % 3 == 0) return false;

        for (int i = 5; i*i<=n; i+=6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    int nextPrime(int n) {
        if (n <= 1) return 2;

        int prime = n;
        bool found = false;
        while (!found) {
            prime++;
            if (isPrime(prime))
                found = true;
        }
        return prime;
    }

    int size;
    vector<int> hashValues;
};

